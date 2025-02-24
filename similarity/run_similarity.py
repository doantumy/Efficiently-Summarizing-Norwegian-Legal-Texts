import sys
import pickle
import os
from collections import Counter
from itertools import product, islice
from multiprocessing import Pool
from typing import Union, List, Tuple

import numpy as np
from loguru import logger

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import spacy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import random

# Disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data_stream(file_name: str, split: str = "train", combined_text: bool = False, 
                     batch_size: int = 64, spacy_model=None, sample_fraction: float = 0.2, 
                     min_text_length: int = 50, random_seed: int = 123, ref_name="reference"):
    """
    Stream and preprocess JSON data from a file in batches for the specified split (train/val/test).
    Supports combined 'summary' and 'text' as a single string or separate processing.

    Args:
        file_name (str): The path to the JSON input file.
        split (str): The split to load ('train', 'val', 'test'). Defaults to 'train'.
        combined_text (bool): Whether to return combined 'summary' and 'text' as a single string per item.
                              Defaults to True.
        batch_size (int): Number of records per batch.
        spacy_model (str): Spacy model for sentence segmentation. Defaults to None.
        sample_fraction (float): Fraction of data to randomly sample (default: 10% or 0.1).
        min_text_length (int): Minimum words for texts to be included.
        random_seed (int): Random seed for reproducibility. Defaults to 123.
        ref_name (str): Name of reference field in dataset. Defaults to "references".

    Yields:
        Union[List[str], Tuple[List[str], List[List[str]], List[str]]]:
            - If combined_text is True, return all concatenated 'summary' and 'text' strings.
            - If combined_text is False, yields batches of tokenized summaries, texts, and references.
    """

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Retrieve data for the specified split
    split_data = data.get(split, [])
    if not split_data:
        logger.warning(f"No data found for split '{split}' in the file.")
        return

    # Randomly sample a fraction of the split data if specified
    num_samples = int(len(split_data) * sample_fraction)
    if sample_fraction < 1.0:
        random.seed(random_seed)  # Set the random seed for reproducibility
        split_data = random.sample(split_data, num_samples)

    if combined_text:
        # Process data as combined 'summary' and 'text'
        combined_data = []
        for item in split_data:
            summary = item.get("summary", "")
            text = item.get("text", [""]) if item.get("text") else ""  # Assume text is a list with one string
            combined_text_item = f"{summary} {text}"
            combined_data.append(combined_text_item)
        yield combined_data
    else:
        # Process data with separate summaries and texts
        for i in range(0, len(split_data), batch_size):
            batch = split_data[i:i + batch_size]
            summaries, texts, refs = [], [], []

            for item in batch:
                summary = item.get("summary", "")
                text = item.get("text", [""])[0] if item.get("text") else ""  # use [0] because text is a list of 1 string
                # text = item.get("text", [""]) if item.get("text") else ""
                ref = item.get(ref_name, "")

                # Check for min_text_length
                if (
                    summary and len(summary.split()) >= min_text_length and
                    text and len(text.split()) >= min_text_length * 3
                ):
                    summary_sents = sentenization(summary, length_limit=10, spacy_model=spacy_model)
                    text_sents = sentenization(text, length_limit=50, spacy_model=spacy_model)
                    if summary_sents and text_sents:
                        summaries.append(summary_sents)
                        texts.append(text_sents)
                        refs.append(ref)
                    else:
                        logger.warning(f"Empty sentenization for id {ref}")

            yield summaries, texts, refs

def save(data, file_path="file_name.pkl"):
    """
    Save file to a pickle file.

    Args:
        data (list[torch.Tensor]): List of data to save.
        file_path (str): Path to the pickle file to save embeddings.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Data saved to {file_path}")


def load(file_path="file_name.pkl"):
    """
    Load data from a pickle file.

    Args:
        file_path (str): Path to the pickle file to load data from.

    Returns:
        list[torch.Tensor]: List of loaded data.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"File loaded from {file_path}")
    return data


class Tokenizer:
    """
    A tokenizer class that initializes and configures a tokenizer model for text tokenization.
    The tokenizer exists on the CPU only.

    Attributes:
        tokenizer (AutoTokenizer): An instance of a Hugging Face AutoTokenizer for text tokenization.

    Methods:
        __call__(text, padding="max_length", truncation=False, max_length=512, add_special_tokens=True, 
                 return_attention_mask=True, return_token_type_ids=False, pad_to_max_length=True, 
                 return_tensors="pt"):
            Tokenizes the input text according to specified parameters and returns the tokenized output.
    """

    def __init__(self, tokenizer=None, default_model="NbAiLab/nb-bert-base"):
        """
        Initializes the Tokenizer with a specified tokenizer model or uses a default model if none is provided.

        Args:
            tokenizer (Union[None, str, AutoTokenizer], optional): A tokenizer name, instance, or None to use the default model.
            default_model (str, optional): Name of the default tokenizer model. Defaults to "NbAiLab/nb-bert-base".
        """
        if tokenizer is None:
            logger.info(f"Using default tokenizer: {default_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(default_model)
        elif isinstance(tokenizer, str):
            logger.info(f"Using tokenizer: {tokenizer}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            logger.info("Using custom tokenizer")
            self.tokenizer = tokenizer

    def __call__(self, text: Union[str, list], padding: Union[str, bool] = False, truncation=False,
                 max_length=512, add_special_tokens=True, return_attention_mask=True,
                 return_token_type_ids=False, return_tensors="pt"):
        """
        Tokenizes input text with specified settings and returns the tokenized result.

        Args:
            text (Union[str, list]): Input text or list of texts to be tokenized.
            padding (Union[str, bool], optional): Padding setting. Defaults to True.
            truncation (bool, optional): Whether to truncate text to fit max_length. Defaults to False.
            max_length (int, optional): Maximum length of tokenized text. Defaults to 512.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.
            return_attention_mask (bool, optional): Whether to return attention mask. Defaults to True.
            return_token_type_ids (bool, optional): Whether to return token type IDs. Defaults to False.
            return_tensors (str, optional): Format of returned tensors. Defaults to "pt".

        Returns:
            dict: Tokenized output with keys such as input_ids, attention_mask, etc., depending on parameters.
        """
        if isinstance(text, str):
            text = [text]
        # logger.info(f"Tokenizing {len(text)} samples")
        return self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_tensors=return_tensors
        )


class TokenIdf:
    """
    A class to calculate Inverse Document Frequency (IDF) values across a collection of documents.
    It handles long documents by splitting and aggregating tokens at the document level.

    Attributes:
        tokenizer (Tokenizer): Instance of the Tokenizer class for text tokenization.
        batch_size (int): Number of documents to process in each batch.
        nthreads (int): Number of threads to use for parallel processing.
        idf_dict (dict): Dictionary to store IDF values for different document groups.

    Methods:
        __call__(documents, group):
            Calculates IDF values for a group of documents and returns an IDF dictionary.
        
        _aggregate_token_presence_batched(documents):
            Aggregates unique tokens across documents in batches to optimize memory usage.
        
        _process_document_chunks(document):
            Tokenizes and aggregates unique tokens across chunks of a single document.
        
        _batched(iterable, batch_size):
            Helper function to yield items in batches.

        save():
            Saves the IDF dictionary to a file.

        load(group):
            Loads a specific IDF dictionary from a file.

        get(group):
            Retrieves the stored IDF dictionary for a specified group.
    """

    def __init__(self, batch_size=128, nthreads=4, default_model=None):
        """
        Initializes the TokenIdf instance with batching and threading settings.

        Args:
            batch_size (int, optional): Number of documents to process in each batch. Defaults to 128.
            nthreads (int, optional): Number of threads for parallel processing. Defaults to 4.
            default_model (str, optional): Default tokenizer model to use. Defaults to None.
        """
        logger.info("Creating token IDF dictionary generator")
        self.tokenizer = Tokenizer(default_model)
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.idf_dict = {}

    def __call__(self, documents: list[str], group: str) -> dict[int, float]:
        """
        Calculates IDF values for a list of documents within a specified group.

        Args:
            documents (list[str]): List of documents for which to calculate IDF.
            group (str): Group name to identify and store the resulting IDF dictionary.

        Returns:
            dict[int, float]: A dictionary mapping token IDs to their IDF values.
        """
        logger.info(f"Calculating IDF dictionary from {len(documents)} documents")
        num_docs = len(documents)

        # Perform batched aggregation for document-level token presence
        doc_level_token_counts = self._aggregate_token_presence_batched(documents)

        # Calculate IDF based on document-level token counts
        idf_count = Counter()
        for tokens in doc_level_token_counts:
            idf_count.update(tokens)

        # Calculate IDF values
        idf_dict = {idx: np.log((num_docs + 1) / (c + 1)) for idx, c in idf_count.items()}
        self.idf_dict[group] = idf_dict
        return idf_dict

    def _aggregate_token_presence_batched(self, documents):
        """
        Aggregates unique tokens per document in batches to improve performance.

        Args:
            documents (list[str]): List of documents to process in batches.

        Returns:
            list[torch.Tensor]: List of tensors, each containing all token IDs for a document.
        """
        # Initialize a nested list to accumulate chunks for each document
        document_tokens = [[] for _ in range(len(documents))]

        # Process documents in batches to control memory usage
        for batch_index, batch in enumerate(self._batched(documents, self.batch_size)):

            if self.nthreads > 0:
                logger.info(f"Processing using parallel with {self.nthreads} threads.")
                with Pool(self.nthreads) as p:
                    batch_tokens = p.map(self._process_document_chunks, batch)
            else:
                logger.info("Processing using sequential.")
                batch_tokens = map(self._process_document_chunks, batch)

            # Add each document's chunk tokens to the appropriate document entry in document_tokens
            start_index = batch_index * self.batch_size
            for i, tokens_in_document_chunks in enumerate(batch_tokens):
                doc_index = start_index + i
                document_tokens[doc_index].extend(tokens_in_document_chunks)

        return document_tokens

    def _process_document_chunks(self, document):
        """
        Tokenizes and aggregates unique tokens across chunks of a single document.

        Args:
            document (str): The document to be split and tokenized.

        Returns:
            set[int]: A set of unique token IDs representing the document.
        """

        def split_document(document, max_words=512):
            """
            Splits a document into smaller chunks that fit within the specified maximum number of words.

            Args:
                document (str): The document to split.
                max_words (int, optional): Maximum chunk length in words. Defaults to 512.

            Returns:
                list[str]: List of document chunks, each containing up to `max_words` words.
            """
            # Split the document by whitespace to get words
            words = document.split()

            # Group words into chunks of `max_words`
            chunks = [" ".join(words[i: i + max_words]) for i in range(0, len(words), max_words)]
            return chunks

        # TODO this can be batched for performance - a chunk contains a list of up to 512 words - this could be a tuple with n number of word chunks
        tokens_in_chunks = [self.tokenizer(chunk)["input_ids"].tolist()[0] for chunk in split_document(document)]
        doc_tokens = set().union(*tokens_in_chunks)

        return doc_tokens

    def _batched(self, iterable, batch_size):
        """
        Yields items from an iterable in batches of a specified size.

        Args:
            iterable (iterable): Iterable to split into batches.
            batch_size (int): Size of each batch.

        Yields:
            list: Batch of items.
        """
        it = iter(iterable)
        while batch := list(islice(it, batch_size)):
            yield batch

    def save(self, file_path: str):
        """
        Saves the IDF dictionary to a file for each group in `idf_dict`.
        """
        os.makedirs(file_path, exist_ok=True)
        logger.info("Saving IDF dictionary")
        for key, value in self.idf_dict.items():
            full_file_path = os.path.join(file_path, f"idf_dict_{key}.pkl")
            with open(full_file_path, "wb") as f:
                pickle.dump(value, f)
            logger.info(f"IDF dictionary for group '{key}' saved to {full_file_path}")

    def load(self, file_path: str, group: str) -> dict[int, float]:
        """
        Loads the IDF dictionary for a specified group from a file.

        Args:
            group (str): The group name of the IDF dictionary to load.

        Returns:
            dict[int, float]: The loaded IDF dictionary for the specified group.
        """
        logger.info(f"Loading IDF dictionary {file_path}")
        with open(f"{file_path}", "rb") as f:
            self.idf_dict[group] = pickle.load(f)
        return self.idf_dict[group]


class Embeddings:
    """
    A class for generating embeddings using a pre-trained model
    Embeddings exists on the GPU or the CPU if no GPU is available
    
    Methods:
        _batched(iterable, batch_size): Helper function to yield items in batches.
    
    Attributes:
        model (AutoModel): model for generating embeddings.
        device (str): device on which the model operates, either 'cuda' or 'cpu'.
    """

    def __init__(self, model="NbAiLab/nb-bert-base", max_length=512, batch_size=64):
        """
        Initializes the Embeddings class with a specified model on the available device.

        Args:
            mex_length (int): max sequence length
            batch_size (int): batch size for model input
            model (str or AutoModel, optional): Model name to load from Hugging Face,
                                                or a custom model instance. Defaults to 'NbAiLab/nb-bert-base'.

        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = Tokenizer(default_model=model)
        self.pad_token_id = self.tokenizer.tokenizer.pad_token_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load specified or default model
        if isinstance(model, str):
            logger.info(f"Loading model: {model}")
            self.model = AutoModel.from_pretrained(model, torch_dtype=torch.float16).to(self.device)
        else:
            logger.info("Using custom model")
            self.model = model.to(self.device)

    def __call__(self, tokens):
        """
        Generates embeddings for the provided tokenized input.

        Args:
            tokens (dict): Tokenized input containing input IDs and attention masks.

        Returns:
            tuple: The output from the model, the attention mask expanded for embeddings,
                   and masked embeddings with attention applied.
        """
        # logger.info("Calculating embeddings")
        tokens = tokens.to(self.device)
        output = self.model(**tokens)
        embeddings = output.last_hidden_state

        # Apply attention mask to embeddings
        att_mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * att_mask

        return output, att_mask, masked_embeddings

    def _batched(self, iterable, batch_size):
        """
        Yields items from an iterable in batches of a specified size.

        Args:
            iterable (iterable): Iterable to split into batches.
            batch_size (int): Size of each batch.

        Yields:
            list: Batch of items.
        """
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]

    def _pad_embeddings(self, embeddings_list):
        """
        Pads a list of embeddings to the same sequence length using pad_token_id or 0.
        """
        max_seq_len = max(emb.shape[1] for emb in embeddings_list)
        # pad_token_id = self.pad_token_id or 0  # Fallback to 0 if pad_token_id is None
        pad_token_id = 0.0
        # Pre-allocate the final padded array
        batch_size = sum(emb.shape[0] for emb in embeddings_list)
        feature_dim = embeddings_list[0].shape[2]
        if any(emb.shape[2] != feature_dim for emb in embeddings_list):
            raise ValueError("Inconsistent feature dimensions in embeddings.")
        # logger.info(f"max_seq_len {max_seq_len} batch_size {batch_size} feature_dim {feature_dim}")
        padded_embeddings = np.full((batch_size, max_seq_len, feature_dim), pad_token_id)

        # Fill in embeddings with padding as needed
        start_idx = 0
        for emb in tqdm(embeddings_list, desc="Padding embeddings"):
            seq_len = emb.shape[1]
            batch_len = emb.shape[0]
            # Insert each embedding at the right position in the pre-allocated array
            padded_embeddings[start_idx:start_idx + batch_len, :seq_len, :] = emb
            start_idx += batch_len

        return padded_embeddings

    def get_embeddings(self, texts, refs):
        """
        Tokenizes and converts a list of documents to embeddings.

        Args:
            texts (list[str]): List of text documents to generate embeddings.
            refs (list[str]): List of reference URL for each document.

        Returns:
            list[torch.Tensor]: List of tensors, each representing the embeddings for a text.
        """
        embeddings_list = []

        for idx, text_list in enumerate(texts):
            document_embeddings = []
            logger.info(f"Processing document id: {idx} - Ref: {refs[idx]}")
            # Process each document in smaller batches
            for text_batch in tqdm(self._batched(text_list, self.batch_size), desc="Processing text batches",
                                   leave=True, position=0):
                tokens = self.tokenizer(text_batch, padding=True, truncation=True,
                                        max_length=self.max_length, return_tensors="pt").to(self.device)

                # Generate embeddings for this batch
                _, _, embeddings = self(tokens)

                # Move embeddings to CPU and store
                document_embeddings.append(embeddings.detach().cpu().numpy())

                # Clear CUDA memory cache
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            # Pad the embeddings to the maximum sequence length within this document
            document_embeddings = self._pad_embeddings(document_embeddings)

            # TODO drop padding tokens by splitting up the result tensor, aka different length per sentences resulting in a list of embedding
            embeddings_list.append(document_embeddings)

        return embeddings_list


class WeightedCosineSimilarity:
    """
    A class for calculating the weighted cosine similarity between two embeddings
    """

    def __init__(self, nthreads: int = 4, lang: str = "no", outout_dir: str = "similarity/output/"):
        """__init__

        Args:
            nthreads (int): Number of processes to use. Defaults to 4.
            lang (str): Language of the dataset. Defaults to "no".
            outout_dir (str): Directory to save output. Defaults to "similarity/output/".
        """
        self.nthreads = nthreads
        self.lang = lang
        self.outout_dir = outout_dir

    def find_max_length(self, data: list[np.ndarray]):
        max_length = 0
        for d in data:
            max_length = max(max_length, d.shape[0])
        return max_length

    def create_matrix(self, data: list[np.ndarray], feature_vector_size: int = 768):
        max_length = self.find_max_length(data)
        matrix = np.zeros((len(data), max_length, feature_vector_size))
        for i, d in enumerate(data):
            matrix[i, :d.shape[0], :] = d
        return matrix

    def create_idf_weight_matrix(self, token_list: list[np.ndarray], idfs: dict[int, float]):
        weights = []
        for tokens in token_list:
            # TODO check what the first element in the token list is (102); (103) is the CLS end token
            # CLS stands for classification and its there to represent sentence-level classification.
            # https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-is-its-encoding-output-important
            # TODO check if we need to fix the value of the first token regarding the idf value of 0
            weight_list = np.asarray([idfs.get(t, 0.0) for t in tokens[0]])  # tokens is List[List[int]]
            weight_list_normalized = np.divide(weight_list,
                                               np.sum(weight_list) + 1e-9)  # add 1e-9 to avoid dividing by zero

            weights.append(weight_list_normalized)
        return weights

    def compute_weighted_similarity(self, i, j, row_matrix, col_matrix_t, row_weights, col_weights):
        try:
            a = row_matrix[i]
            b = col_matrix_t[j]
        except IndexError:
            return i, j, None

        c = np.matmul(a, b)
        precision = np.max(c, axis=1)
        row_scale = row_weights[i]
        word_precision = np.multiply(precision[:row_scale.shape[0]], row_scale)
        P = np.sum(word_precision)
        recall = np.max(c, axis=0)
        col_scale = col_weights[j]
        word_recall = np.multiply(recall[:col_scale.shape[0]], col_scale)
        R = np.sum(word_recall)
        F = 2 * P * R / (P + R)
        return i, j, F

    def run_weighted_similarity_per_document(self,
                                             embedding: tuple[list[np.ndarray], list[np.ndarray]] = None,
                                             token: tuple[list[np.ndarray], list[np.ndarray]] = None,
                                             idf: dict[int, float] = None,
                                             grid_size=128, feature_vector_size=768):
        """
        Compute the weighted similarity between two sets of embeddings
        Set one contains the summary sentences, set two contains the text sentences

        :param embedding: tuple of two lists of np.ndarray
        :param token: tuple of two lists of np.ndarray
        :param idf: dict[int, float]
        :param grid_size: grid size for processing
        :param feature_vector_size: feature vector size
        :return: np.ndarray of shape (len(embedding[0]), len(embedding[1])) with the weighted similarities
        """

        # all_embeddings = documents[
        #      (summary_sentences[embeddings=np.ndarray[len(tokens),768]],
        #       text_sentences[embeddings=np.ndarray[len(tokens),768]])
        # ]
        # all_tokens = documents[
        #     (summary_sentences[len(tokens)],
        #      text_sentences[len(tokens)])
        # ]
        # idfs = dict[tokenID, float] # idf values for each token

        # similarity matrix
        sims = np.zeros((len(embedding[0]), len(embedding[1])))

        # summary sentences as row embeddings
        row_embeddings = embedding[0]
        rows = len(row_embeddings)
        row_matrix = self.create_matrix(row_embeddings, feature_vector_size)

        # normalize the embeddings
        row_normalizer = np.linalg.norm(row_matrix, axis=-1, keepdims=True)
        row_matrix = np.nan_to_num(np.divide(row_matrix, row_normalizer,
                                             out=np.zeros_like(row_matrix), where=row_normalizer != 0))
        row_tokens = token[0]
        row_weights = self.create_idf_weight_matrix(row_tokens, idf)

        # for extended document, this could be used for parallel processing
        grid_steps = list(range(0, len(embedding[1]), grid_size))

        # text sentences as column embeddings
        for _, col in tqdm(enumerate(grid_steps), desc=f"Column position",
                           total=len(grid_steps), position=1, colour='blue', leave=False):
            col_embeddings = embedding[1][col:col + grid_size]
            col_matrix = self.create_matrix(col_embeddings, feature_vector_size)
            # normalize the embeddings
            col_normalizer = np.linalg.norm(col_matrix, axis=-1, keepdims=True)
            col_matrix = np.nan_to_num(np.divide(col_matrix, col_normalizer,
                                                 out=np.zeros_like(col_matrix), where=col_normalizer != 0))
            col_matrix_t = col_matrix.transpose((0, 2, 1))
            col_tokens = token[1][col:col + grid_size]
            col_weights = self.create_idf_weight_matrix(col_tokens, idf)

            # full iteration of the local grid between the summary and local grid text sentences
            full_iter = list(product(range(rows), range(grid_size)))
            Fs = [self.compute_weighted_similarity(i, j, row_matrix, col_matrix_t, row_weights, col_weights)
                  for i, j in tqdm(full_iter, total=rows * grid_size, leave=False,
                                   position=2, desc=f"Computing similarities", colour='red')]

            # update the similarity matrix
            for i, j, F in Fs:
                if F is None:
                    continue
                sims[i, col + j] = F

        return sims

    def process_similarity(self, idx, ref, sum_embedding, txt_embedding, sum_token, txt_token, idf_load):
        """
        Compute similarity matrix for a single document.

        Args:
            idx (int): Document index.
            ref (str): Reference ID for naming outputs.
            sum_embedding (np.ndarray): Summary embeddings.
            txt_embedding (np.ndarray): Text embeddings.
            sum_token (np.ndarray): Summary tokens.
            txt_token (np.ndarray): Text tokens.
            idf_load (dict): IDF values for weighting.

        Returns:
            tuple: (int, bool) where bool indicates success or failure.
        """
        try:
            # Compute similarity matrix
            document_similarity_matrix = self.run_weighted_similarity_per_document(
                (sum_embedding, txt_embedding),
                (sum_token, txt_token),
                idf_load
            )

            # Save similarity histogram
            save_similarity_histogram(
                document_similarity_matrix,
                doc_id=f"idx_{idx}_ref_{ref.split('/')[-1]}",
                bins=50,
                color='green',
                alpha=0.5,
                output_dir=f"similarity/output/{self.lang}_figures/"
            )

            return idx, True
        except Exception as e:
            logger.error(f"Error processing document {idx}: {e}")
            return idx, False

    def run_parallel_similarity(self, refs, embeddings, tokens, idf_load):
        """
        Run similarity computation in parallel using multiprocessing.

        Args:
            refs (list): List of document references for naming histograms.
            embeddings (tuple): Tuple containing summary and text embeddings.
            tokens (tuple): Tuple containing summary and text tokens.
            idf_load (dict): IDF values for weighting.
        """
        # Unpack embeddings and tokens
        sum_embeddings, txt_embeddings = embeddings
        sum_tokens, txt_tokens = tokens

        # Prepare tasks
        tasks = [
            (idx, refs[idx], sum_embeddings[idx], txt_embeddings[idx], sum_tokens[idx], txt_tokens[idx], idf_load)
            for idx in range(len(refs))
        ]

        # Parallel processing with multiprocessing.Pool
        with Pool(self.nthreads) as pool:
            results = list(
                tqdm(pool.starmap(self.process_similarity, tasks),
                     total=len(tasks),
                     desc="Processing similarity matrices")
            )

        # Handle results
        for idx, success in results:
            if not success:
                logger.warning(f"Failed to process document {idx}")


class TokenSent:
    """
    Tokenized sentence
    """

    def __init__(self, batch_size=128, max_length=512, default_model="NbAiLab/nb-bert-base"):
        """
        Initializes the TokenIdf instance with batching and threading settings.

        Args:
            batch_size (int): Number of documents to process in each batch. Defaults to 128.
            default_model (str): Default tokenizer model to use.
            save_dir (str): Directory to save tokenized files. Defaults to "/no_tok_output/sum/".
        """
        logger.info("Creating token IDF dictionary generator")
        self.tokenizer = Tokenizer(default_model)
        self.batch_size = batch_size
        self.max_length = max_length

    def token_list(self, documents, truncation=False):
        """Tokenizes the text using Spacy and returns a list of tokens

        Args:
            text (str): Document text
        """
        token_list = []

        for _, doc in enumerate(tqdm(documents, desc="Processing tokenization sentence", leave=True, position=0)):
            doc_tokens = []
            for sentence in doc:
                tokens = self.tokenizer(sentence, truncation=truncation, padding=False, max_length=self.max_length)
                doc_tokens.append(tokens["input_ids"].detach().cpu().numpy())
            token_list.append(doc_tokens)
        return token_list


def sentenization(text, length_limit=50, spacy_model="nb_core_news_sm", max_length=3_000_000):
    """Sentence segmentation using Spacy

    Args:
        text (str): Document text
        length_limit (int): Min number of characters per sentence
        spacy_model (str): Spacy model to use for sentence segmentation. Defaults to "nb_core_news_sm".

    Returns:
        List[str]: A list of sentences
    """
    if not spacy_model:
        raise ValueError("Spacy NLP model is required for sentenization.")

    nlp = spacy.load(spacy_model)
    nlp.max_length=max_length
    doc = nlp(text)
    return [sent.text for sent in doc.sents if
            len(sent.text) >= length_limit]  # Only keep sentences >= length_limit characters long


def save_similarity_histogram(document_similarity_matrix, doc_id, bins=30, color='green', alpha=0.7,
                              output_dir="similarity/output/no_figure/", lang="no"):
    """
    Generates and saves a histogram of similarity values for a document's similarity matrix.

    Args:
        document_similarity_matrix (np.ndarray): 2D array of similarity values for the document.
        doc_id (str): Unique identifier for the document, using doc ID from URL reference.
        bins (int): Number of bins for the histogram. Defaults to 30.
        color (str): Color of the histogram. Defaults to 'blue'.
        alpha (float): Transparency level of the histogram. Defaults to 0.7.
        output_dir (str): Directory where the figure will be saved. Defaults to "similarity/output/figure".
        lang (str): Language of the document. Defaults to "no".
    """
    # Ensure output directory exists
    output_path = f"{output_dir}_{lang}_figures/"
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Creating output directory {output_path}")
    # Flatten the similarity matrix to a 1D array
    similarity_values = document_similarity_matrix.flatten()

    # Plot histogram
    plt.figure()
    plt.hist(similarity_values, bins=bins, color=color, alpha=alpha)
    plt.title(f"Similarity Histogram - {doc_id}")
    plt.xlabel("Similarity Value")
    plt.ylabel("Frequency")

    # Save figure with document ID in the filename
    file_path = os.path.join(output_path, f"sim_hist_{doc_id}.png")
    plt.savefig(file_path)
    plt.close()  # Close the figure to free up memory
    logger.info(f"Histogram saved for Document {doc_id} at {file_path}")

def process_tokenidf_documents(file_name, batch_size, sample_fraction, random_seed, combined_text, split, ref_name):
    """
    Process a JSON file containing documents and their summaries, calculate IDF scores, and save the results.

    Args:
        file_name (str): Path to JSON input file.
        batch_size (int): Number of examples per batch.
        sample_fraction (float): Fraction of documents to sample for processing.
        ref_name (str): Name of the reference

    Returns:
        List[str]: Combined text of the sampled documents.
    """
    combined_texts = []
    for _, combined_text in enumerate(tqdm(load_data_stream(file_name = file_name, 
                                                            split = split, 
                                                            combined_text = combined_text, 
                                                            batch_size=batch_size,
                                                            sample_fraction = sample_fraction, 
                                                            random_seed = random_seed,
                                                            ref_name=ref_name), desc="Processing IDF for documents")):
        combined_texts.extend(combined_text)
    return combined_texts

def process_embeddings_and_similarity(file_name, batch_size, model, 
                                      spacy_model, idf_load, 
                                      random_seed, sample_fraction, 
                                      combined_text, split, ref_name,
                                      output_dir="similarity/output/train", lang="no"):
    """
    Stream process data, compute embeddings, and calculate similarity in-memory.

    Args:
        file_name (str): Path to JSON input file.
        batch_size (int): Number of examples per batch.
        model (str): Pre-trained model for embeddings.
        spacy_model (str): Spacy model.
        output_dir (str): Path to output similarity matrix.
        lang (str): Language of the documents. Defaults to "no".
        ref_name (str): Name of the reference in data.
    """

    # Create folder for output similarity matrix if doens't exist
    output_path = f"{output_dir}_{lang}_sim_matrix/"
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Creating output folder {output_path}")

    emb = Embeddings(model=model, max_length=512, batch_size=batch_size)
    wcs = WeightedCosineSimilarity(lang=lang)
    tks = TokenSent(batch_size=batch_size, max_length=512, default_model=model)

    for batch_idx, (summaries, texts, refs) in enumerate(
            tqdm(load_data_stream(file_name, batch_size=batch_size, 
                                  spacy_model=spacy_model, 
                                  sample_fraction=sample_fraction, 
                                  random_seed=random_seed, 
                                  combined_text=combined_text, 
                                  split=split, ref_name=ref_name),
                 desc="Streaming data")
    ):
        # Compute embeddings for summaries and texts
        sum_embeddings = emb.get_embeddings(texts=summaries, refs=refs)
        txt_embeddings = emb.get_embeddings(texts=texts, refs=refs)

        # Tokenizing summaries and texts
        sum_tokens = tks.token_list(documents=summaries, truncation=True)
        txt_tokens = tks.token_list(documents=texts, truncation=True)

        # Calculate similarity matrices and process results
        with open(f"similarity/{lang}_error.txt", "a") as error_log:
            for idx, (sum_emb, txt_emb, ref,
                    sum_token, txt_token) in enumerate(zip(sum_embeddings, txt_embeddings, refs,
                                                            sum_tokens, txt_tokens)):
                try:
                    document_similarity_matrix = wcs.run_weighted_similarity_per_document(
                        (sum_emb, txt_emb), (sum_token, txt_token), idf_load
                    )
                    if not np.any(np.isnan(document_similarity_matrix)):
                        save(data=document_similarity_matrix,
                            file_path=f"{output_path}batch_{batch_idx}_doc_{idx}_ref_{ref.split('/')[-1]}.pkl")
                        
                        # Save similarity histgram
                        save_similarity_histogram(
                            document_similarity_matrix,
                            doc_id=f"batch_{batch_idx}_doc_{idx}_ref_{ref.split('/')[-1]}",
                            bins=50,
                            color='green',
                            alpha=0.5,
                            output_dir=output_dir,
                            lang=lang
                        )
                    else:
                        logger.warning(f"Sim matrix is nan. Ref: {ref}")
                except ValueError as e:
                    logger.warning(f"Error: docID {ref}")
                    error_log.write(f"{ref}\n")

        # Discard embeddings and similarity matrices to free memory
        del sum_embeddings, txt_embeddings, sum_tokens, txt_tokens


if __name__ == "__main__":
    # data_file = "data/processed_data/test.json"
    # Norwegian LovData
    # default_model = "NbAiLab/nb-bert-base"
    # data_file = "data/processed_data/processed_data_wo_segment.json"
    # spacy_model = "nb_core_news_sm"
    # lang="no"
    # ref_name = "reference"

    # Norwegian ALL LovData
    default_model = "NbAiLab/nb-bert-base"
    data_file = "data/all_processed_data/all_data_wo_seg_wo_sent.json"
    spacy_model = "nb_core_news_sm"
    lang="no_all"
    ref_name = "filename"

    # # Australian LovData
    # default_model = "google-bert/bert-base-uncased"
    # data_file = "data/au_data_processed/au_processed_data_wo_segment.json"
    # spacy_model = "en_core_web_sm" # python -m spacy download en_core_web_sm
    # lang="au"
    # ref_name = "reference"


    splits = ["val", "test", "train"]
    batch_size = 64
    sample_fraction = 1.0
    random_seed = 123
    for split in splits:
        logger.info(f"Processing TokenIDF for {split} split - sample_fraction {sample_fraction} - seed {random_seed}...")
        documents_list = process_tokenidf_documents(file_name=data_file, 
                                                    split=split, 
                                                    sample_fraction=sample_fraction, 
                                                    random_seed=random_seed, 
                                                    combined_text=True, 
                                                    batch_size=batch_size, ref_name=ref_name)

        idf_gen = TokenIdf(default_model=default_model)
        idf_load = idf_gen(documents_list, "lov")

        logger.info(f"Processing Similarity matrix for {split} split - sample_fraction {sample_fraction} - seed {random_seed}...")
        output_dir = f"similarity/output/{split}"
        process_embeddings_and_similarity(
            file_name=data_file,
            batch_size=batch_size,
            model=default_model,
            spacy_model=spacy_model,
            idf_load=idf_load,
            sample_fraction=sample_fraction,
            random_seed=random_seed,
            output_dir=output_dir,
            combined_text=False,
            split=split, lang=lang, ref_name=ref_name
        )