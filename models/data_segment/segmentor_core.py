import abc
import multiprocessing
from functools import partial

import nltk
from loguru import logger
from rouge import Rouge
from rouge.rouge_score import Ngrams, f_r_p_rouge_n
from typing import Dict, List, Union

from stanza import DownloadMethod

from utils.tools import download_nltk
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore
from bert_score import score as BERTScore
import stanza


# Download the Norwegian model, only need the first time
# stanza.download('no')
# Initialize the pipeline


def _rouge_(hypo, target, func):
    return func(preds=hypo, target=target)["rouge1_fmeasure"].item()


class SourceSimilarityScoreSplitter(object):
    def __init__(self, metric: str = None, input_max_token: int = 250):
        """ Matching given inputs together using either ROUGE-1 score or BERTScore embeddings.

        Args:
            config (Configure): Configuration object.
            metric (str, optional): name of metric, can be 'rouge-1' or 'bertscore'. Defaults to None.

         Returns:
            List[str]: A list of paragraphs after preprocessing.
        """
        download_nltk()
        self.metric = metric
        self.input_max_token = input_max_token
        # Pipeline for sentence-level tokenization for Norwegian
        nlp = stanza.Pipeline('no', download_method=DownloadMethod.REUSE_RESOURCES)
        self.nlp = nlp
        self.nthreads = multiprocessing.cpu_count()

    def paragraph_merging(self, paragraph_list: List[str]) -> tuple[List[str], int]:
        """Preprocessing a list of sentences in a paragraph for Norwegian

        Args:
            paragraph_list (List[List[str]]): paragraph list of sentences
            P = ["sentence 1", "sentence 2. sentence 3", "sentence 4"]

        Returns:
            Union[List[List[str]], int, int]:
            - A list of sentence-level tokenization: ["sentence 1", "sentence 2", "sentence 3", "sentence 4"]
            - Total tokens across all sentences in the paragraph: word-level tokens
        """
        total_tokens = 0
        sentence_list = []

        for item in paragraph_list:
            doc = self.nlp(item)
            sentence_list.extend([sentence.text for sentence in doc.sentences])
            total_tokens += len(item.split())
        return sentence_list, total_tokens

    @staticmethod
    def compute_rouge1_score(rouge_func, nthreads, hypothesis, reference):
        """Compute ROUGE-1 score between two strings.

        Args:
            reference (list): reference string.
            hypothesis (list): hypothesis string.

        Returns:
            score [float]: rouge-1 score
        """

        with multiprocessing.Pool(nthreads) as p:
            score = p.starmap(partial(_rouge_, func=rouge_func), zip(hypothesis, reference))
        return score

    def compute_bertscore(self, hypothesis, reference, model_type="bert-base-multilingual-cased"):
        """Computes the BERTScore F1 score between candidate and reference strings.

        Args:
            hypothesis (List[str]): List of candidate strings.
            reference (List[str]): List of reference strings.
            model_type (str, optional): Model name to get the embeddings from. Defaults to "bert-base-multilingual-cased".

        Returns:
            score [float]: BERTScore F1 score.
        """
        P, R, F1 = BERTScore(hypothesis, reference, lang="nb", model_type=model_type, rescale_with_baseline=False)
        return F1.numpy().tolist()

    def similar_sentences_segment(self, data: Union[List[str], Dict[str, Dict[str, List[str]]]] = None) -> List[str]:
        """Calculate the similarity score for two consecutive sentences in a paragraph
        and return a list of new segments that are within the token limit.
        For example:
        Case 1: If the total number of tokens in the paragraph <= token limit, we keep the paragraph as it is.
        Case 2: The total number of tokens in the paragraph > token limit, we break the paragraph into segments at sentence-level.
            Paragraph P = ["sentence 1", "sentence 2", "sentence 3", "sentence 4"]
            score(p_1, p_2) = sim(sentence 1, sentence 2)
            score(p_2, p_3) = sim(sentence 2, sentence 3)
            score(p_3, p_4) = sim(sentence 3, sentence 4)
            Normalize scores list
            Find the min(scores) and break the paragraph at the index of the min(scores)
            Returns a new paragraph list with 2 sequences.
            Go back to Case 1 and repeat the process until the total number of tokens in the paragraph <= token limit.

        Args:
            data (Union[List[str], Dict[str, Dict[str, List[str]]]]): Input data. Defaults to None.
            Data can be a list of strings or a dictionary with KAPITTEL and sub_KAPITTEL (with a list of items)

        Returns:
            List[str]: A list contains all sentences in a paragraph such that:
            - each element (one or more sentences merged together in a segment) in it is within the token limit 
        """
        # WARNING THIS CODE HAS SIDE EFFECTS
        all_segment_list = []

        # Initialize ROUGE evaluator
        rouge = ROUGEScore()

        def break_similar_sentences(paragraph_list: List[str], similarities: List[float], depth: int = 0, max_depth: int = 900):
            """[ RECURSIVE FUNCTION ] CASE 2: Calculate the similarity score for two consecutive sentences in a paragraph
            and return a list of new segments that are within the token limit.
            
                Paragraph P = ["sentence 1", "sentence 2", "sentence 3", "sentence 4"]
                score(p_1, p_2) = sim(sentence 1, sentence 2)
                score(p_2, p_3) = sim(sentence 2, sentence 3)
                score(p_3, p_4) = sim(sentence 3, sentence 4)
                Normalize scores list
                Find the min(scores) and break the paragraph at the index of the min(scores)
                Returns a new paragraph list with 2 sequences.
                Go back to Case 1 and repeat the process until the total number of tokens in the paragraph <= token limit.

            Args:
                paragraph_list (List[str]): A list contains all sentences in a paragraph.
                similarities (List[float]): A list contains all similarity scores of consecutive pairs.
                depth (int): Current depth of recursion. Defaults to 0.
                max_depth (int): Maximum allowed recursion depth. Defaults to 500.

            Returns:
                List[str]: A list contains all sentences in a paragraph such that:
                - each element (one or more sentences merged together in a segment) in it is within the token limit 
            """
            # Initialize a list to store new segments
            nonlocal all_segment_list
            # logger.info(f"Depth: {depth}")
            # Base case: If there's only one sentence in the list, add it as a single segment
            if len(paragraph_list) == 1:
                all_segment_list.extend(paragraph_list)
                return

            # Base case: If total tokens in the paragraph list are within the limit
            total_tokens = sum(len(sentence.split()) for sentence in paragraph_list)
            if total_tokens <= self.input_max_token:
                all_segment_list.append(" ".join(paragraph_list))
                return
            
            # Check if similarities are empty; if so, add the paragraph as is to prevent infinite recursion
            if not similarities:
                all_segment_list.append(" ".join(paragraph_list))
                return
            
            # If the length of similarities is less than 1, it indicates there's no more splitting possible
            if len(similarities) < 1:
                all_segment_list.append(" ".join(paragraph_list))
                return
            
            # Depth guard: Check if we've exceeded the maximum depth
            if depth > max_depth:
                logger.info(f"Maximum recursion depth reached: {depth}. Exiting recursion to prevent overflow.")
                all_segment_list.append(" ".join(paragraph_list))  # Add the remaining part as a single segment
                return
            
            # Find the index of the minimum similarity score
            min_index = similarities.index(min(similarities))

            # Ensure that the segment to be split has more than one sentence
            if min_index == len(paragraph_list) - 1:
                # If `min_index` is at the end, splitting doesn't make sense; this prevents infinite loops
                all_segment_list.append(" ".join(paragraph_list))
                return
            
            # Split the list into two segments
            segment1 = paragraph_list[:min_index + 1]
            segment2 = paragraph_list[min_index + 1:]

            # Recursively break the new segments if they are over the token limit, increasing the depth
            break_similar_sentences(paragraph_list=segment1,
                                    similarities=similarities[:min_index],
                                    depth=depth + 1, 
                                    max_depth=max_depth)
            break_similar_sentences(paragraph_list=segment2,
                                    similarities=similarities[min_index + 1:],
                                    depth=depth + 1,
                                    max_depth=max_depth)

        # Case 1: If the total number of tokens in the paragraph <= token limit, we keep the paragraph as it is.
        # Case 2: The total number of tokens in the paragraph > token limit, we break the paragraph into segments at sentence-level.    
        # Handle cases where data is a List
        if isinstance(data, List):
            paragraph_list, total_tokens = self.paragraph_merging(data)
            if total_tokens <= self.input_max_token:
                all_segment_list.append(" ".join(paragraph_list))
            else:
                # Calculate similarity scores for all consecutive sentence pairs
                all_similarities = []
                hypothesis = paragraph_list[:-1]
                reference = paragraph_list[1:]

                if self.metric == "bertscore":
                    # TODO #50 vectorize or parallelize this
                    score = self.compute_bertscore(hypothesis=hypothesis, reference=reference)
                    all_similarities = score
                if self.metric == "rouge1":
                    # TODO #50 vectorize or parallelize this
                    score = self.compute_rouge1_score(rouge_func=rouge, nthreads=self.nthreads,
                                                      hypothesis=hypothesis, reference=reference)
                    all_similarities = score
                break_similar_sentences(paragraph_list=paragraph_list,
                                        similarities=all_similarities)

        # Handle cases where data is a dictionary (KAPITTEL and sub_KAPITTEL)
        if isinstance(data, Dict):
            for _, kapittel_data in data.items():
                for _, sub_kapittel_data in kapittel_data.items():
                    paragraph_list, total_tokens = self.paragraph_merging(sub_kapittel_data)
                    if total_tokens <= self.input_max_token:
                        all_segment_list.append(" ".join(paragraph_list))
                    else:
                        # Calculate similarity scores for all consecutive sentence pairs
                        all_similarities = []
                        hypothesis = paragraph_list[:-1]
                        reference = paragraph_list[1:]

                        if self.metric == "bertscore":
                            # TODO #50 vectorize or parallelize this
                            score = self.compute_bertscore(hypothesis=hypothesis, reference=reference)
                            all_similarities = score
                        if self.metric == "rouge1":
                            # TODO #50 vectorize or parallelize this
                            score = self.compute_rouge1_score(rouge_func=rouge, nthreads=self.nthreads,
                                                              hypothesis=hypothesis, reference=reference)
                            all_similarities = score
                        break_similar_sentences(paragraph_list=paragraph_list,
                                                similarities=all_similarities)

        return all_segment_list


class SourceParagraphSplitter(object):
    """ Segments the source document into paragraphs

    Args:
        object (object): input object
    """

    def __init__(self, max_tokens_per_segment):
        self.max_tokens = max_tokens_per_segment

    def segment_one_sample(self, document: Union[List[str], Dict[str, Dict[str, List[str]]]]) -> List[str]:
        """Segment the document into paragraph lavel. 
        We go through each KAPITTEL -> Sub_KAPITTEL -> PARAGRAPH -> each item inside PARAGRAPH list

        Args:
            document (Union[List[str], dict]): input document can be a list of sentences or a dict
            max_tokens (int): maximum number of tokens allowed per segment

        Returns:
            List[str]: A list of paragraphs
        """
        paragraphs = []
        current_paragraph = ""

        def add_paragraph(next_paragraph):
            """Nested function to add the next paragraph to the current paragraph.
            If the current paragraph and the next paragraph are within the word limit, 
            add the next paragraph to current paragraph.

            Args:
                next_paragraph (str): next paragraph to be added to the current paragraph
            """
            # nonlocal referes to the same current_paragraph inside and the outer function
            nonlocal current_paragraph
            # Check if the current paragraph and the next paragraph are within the word limit
            if len(current_paragraph.split()) + len(next_paragraph.split()) <= self.max_tokens:
                # If within the word limit, add the next paragraph to current paragraph
                current_paragraph += " " + next_paragraph.strip()
            else:
                if current_paragraph:
                    paragraphs.append(current_paragraph.strip())
                # Start a new current_paragraph with the next paragraph
                current_paragraph = next_paragraph.strip()

        if isinstance(document, dict):
            for _, sub_kapittel_dict in document.items():
                for _, paragraph_list in sub_kapittel_dict.items():
                    combined_paragraph = ""
                    for paragraph in paragraph_list:
                        if len(combined_paragraph.split()) + len(paragraph.split()) <= self.max_tokens:
                            combined_paragraph += " " + paragraph.strip()
                        else:
                            if combined_paragraph:
                                add_paragraph(combined_paragraph.strip())
                            combined_paragraph = paragraph.strip()
                    if combined_paragraph:
                        add_paragraph(combined_paragraph.strip())
                # Append the last paragraph if it is not empty
                if current_paragraph:
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""  # Reset for the next sub_kapittel

        if isinstance(document, list):
            for paragraph in document:
                add_paragraph(paragraph)
            if current_paragraph:
                paragraphs.append(current_paragraph.strip())

        return paragraphs


class SourceSplitterCore(object):
    """Segments a list of text samples into smaller subgroups based on max token count.

    Args:
        object (any): input object
    """

    def __init__(self, max_tokens_per_segment):
        self.max_tokens = max_tokens_per_segment

    def segment_one_sample(self, sentences) -> list:
        """Segmenting a list of sentences into smaller subgroups based on max token count.
        We group sentences that are shorter than max_tokens into a segment containing a list of sentence.
        If a sentence is longer than max_tokens it will be added as a new segment.

        Args:
            sentences (list): the input is a list of sentences
            max_tokens (int): maximum number of tokens allowed per segment

        Returns:
            segments (list): The output is a list of segments
        """
        # store all the segments
        segments = []
        # store sentences for the current segment
        current_segment = []
        current_token_count = 0

        for sentence in sentences:
            cleaned_sentence = sentence.replace('\n', ' ').replace('\r', '').replace('@@ ', '')
            sentence_length = len(cleaned_sentence.split(' '))

            if current_token_count + sentence_length > self.max_tokens:
                if current_token_count != 0:
                    segments.append(current_segment)
                current_segment = [cleaned_sentence]
                current_token_count = sentence_length
            else:
                current_segment.append(cleaned_sentence)
                current_token_count += sentence_length

        if current_segment:
            # The current segment is added into the segments list before starting new segments.
            segments.append(current_segment)

        return segments


class TargetSplitterCore(object):
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        self.rouge = Rouge()

    def _get_rouge_from_ngram(self, reference_ngrams: Ngrams, evaluated_ngrams: Ngrams) -> dict:
        """

        :param reference_ngrams: summary sentence as ngram as collection of previous fitting summary sentences
        :param evaluated_ngrams: subgroup sentence as ngram
        :return:
        """
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)
        return f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count)

    def _text_to_ngrams(self, text, n=1):
        ngrams = list(nltk.ngrams(nltk.word_tokenize(text), n))
        return Ngrams(ngrams)

    # This function is faster than seg_based_on_rouge because it uses the ngrams to computer rouge rather than text.
    def fast_rouge(self, sou, tar=None, name=None, verbose=False):
        if len(sou) == 3:
            # fix auto unpacking of multiprocessing map
            sou, tar, name = sou
        cur_new = ''
        cur_ngram = Ngrams()
        best_score = 0
        best_sents = []

        # use ngram to represent each text subgroup
        sou = self._text_to_ngrams(sou)
        # seg = for each sentence in tar, get the ngram representation and the sentence
        seg = [(x, self._text_to_ngrams(x), i) for i, x in enumerate(nltk.sent_tokenize(tar))]

        tot_len = len(seg)
        for i in range(min(self.max_length, tot_len)):
            # for each sentence in the summary, get rouge score of the ngram overlab
            scores = [(
                x,
                self._get_rouge_from_ngram(cur_ngram.union(seg_ngram), sou),
                i)
                for x, seg_ngram, i in seg
            ]
            # get best fitting sentence from the summary to the subgroup text
            best_seg = max(scores, key=lambda x: x[1]['f'])
            seg = [x for x in seg if x[2] != best_seg[2]]  # remove dup
            # extend the current summary text with the currently best fitting summary sentence to the subgroup
            # this reorders the summary sentences
            cur_new += ' ' + best_seg[0]
            cur_ngram = self._text_to_ngrams(cur_new)
            # calculate a new rouge score from the current summary text to the subgroup
            cur_score = self._get_rouge_from_ngram(cur_ngram, sou)['f']
            # if the new rouge score is better than the previous best score,
            # update the best score and the best sentences
            # continue until the summary text does not improve based on the current subgroup
            if cur_score > best_score:
                best_score = cur_score
                best_sents.append(best_seg)
            else:
                break

        if verbose:
            logger.info("id:", name, "input/output:", tot_len, len(best_sents), "best:", best_score)
        best_string = list(set((x[0], x[2]) for x in best_sents))
        # revert reordering of the summary sentences as best fitting to the subgroup
        best_string.sort(key=lambda x: x[1])
        # create final summary text to subgroup
        best_string = ' '.join([x[0] for x in best_string])

        return best_sents, best_string

    def seg_based_on_rouge(self, sou, tar, name=None, verbose=False) -> (list, str):
        cur_new = ''
        best_score = 0
        best_sents = []
        seg = [(x, i) for i, x in enumerate(nltk.sent_tokenize(tar))]
        tot_len = len(seg)
        for i in range(min(self.max_length, tot_len)):
            scores = [(x, self.rouge.get_scores(cur_new + ' ' + x, sou), i) for x, i in seg]
            scores.sort(key=lambda x: -x[1][0]['rouge-1']['f'])
            cur_new += scores[0][0] + ' '
            seg = [x for x in seg if x[1] != scores[0][2]]  # remove dup
            cur_score = self.rouge.get_scores(cur_new, sou)[0]['rouge-1']['f']
            if cur_score > best_score:
                best_score = cur_score
                best_sents.append(scores[0])
            else:
                break

        if verbose:
            logger.info("id:", name, "input/output:", tot_len, len(best_sents), "best:", best_score)
        best_string = list(set((x[0], x[2]) for x in best_sents))
        best_string.sort(key=lambda x: x[1])
        best_string = ' '.join([x[0] for x in best_string])

        return best_sents, best_string
