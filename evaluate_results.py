from loguru import logger
from torchmetrics.text.rouge import ROUGEScore
import multiprocessing
from functools import partial
from bert_score import score as BERTScore
from typing import List
import numpy as np

class EvaluateResults(object):
    def __init__(self, metric: str=None, hypothesis_file=None, reference_file=None):
        self.metric = metric
        self.hypothesis_file = hypothesis_file
        self.reference_file = reference_file
        self.nthreads = multiprocessing.cpu_count()


    def file_to_list(self, file_name):
        """Read input file as List[str]

        Returns:
            data: List of items
        """
        with open(file_name, "r") as file:
            lines = [l.strip() for l in file.readlines()]
        return lines


    def compute_rouge1_score(self, hypothesis, reference):
        """Compute ROUGE scores between two strings.

        Args:
            reference (List[str]): reference string.
            hypothesis (List[str]): hypothesis string.

        Returns:
            score [float]: rouge scores
        """

        rouge = ROUGEScore()
        scores = rouge(preds=hypothesis,
                            target=reference)
        return scores

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
        average_f1 = np.mean(F1.numpy())
        return average_f1


    def calculate(self):
        # Read input files
        hypothesis = self.file_to_list(file_name=self.hypothesis_file)
        reference = self.file_to_list(file_name=self.reference_file)
        # Compute scores based on the selected metric
        bert_score = self.compute_bertscore(hypothesis=hypothesis, reference=reference)
        logger.info(f"BERTScore F1 score: {bert_score}")
        rouge_score = self.compute_rouge1_score(hypothesis=hypothesis, reference=reference)
        logger.info(f"ROUGE scores: {rouge_score}")
        return bert_score, rouge_score


# Classic NorT5-base-model
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/classic/generated_summaries_base.txt", reference_file="./output/results/classic/test.target").calculate()
# BERTScore F1 score: 0.714982807636261
# ROUGE scores: {'rouge1_fmeasure': tensor(0.3012), 'rouge1_precision': tensor(0.2640), 'rouge1_recall': tensor(0.4402), 'rouge2_fmeasure': tensor(0.1478), 'rouge2_precision': tensor(0.1283), 'rouge2_recall': tensor(0.2241), 'rougeL_fmeasure': tensor(0.2107), 'rougeL_precision': tensor(0.1844), 'rougeL_recall': tensor(0.3132), 'rougeLsum_fmeasure': tensor(0.2785), 'rougeLsum_precision': tensor(0.2445), 'rougeLsum_recall': tensor(0.4068)}

# Classic NorT5-large-model
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/classic/generated_summaries_large.txt", reference_file="./output/results/classic/test.target").calculate()
# BERTScore F1 score: 0.7754284739494324
# ROUGE scores: {'rouge1_fmeasure': tensor(0.3976), 'rouge1_precision': tensor(0.4735), 'rouge1_recall': tensor(0.3986), 'rouge2_fmeasure': tensor(0.2179), 'rouge2_precision': tensor(0.2554), 'rouge2_recall': tensor(0.2237), 'rougeL_fmeasure': tensor(0.3101), 'rougeL_precision': tensor(0.3678), 'rougeL_recall': tensor(0.3134), 'rougeLsum_fmeasure': tensor(0.3742), 'rougeLsum_precision': tensor(0.4461), 'rougeLsum_recall': tensor(0.3753)}

# summn_base
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/summn/test.base.hypo", reference_file="./output/results/summn/test.target").calculate()
# BERTScore F1 score: 0.7096424698829651
# ROUGE scores: {'rouge1_fmeasure': tensor(0.2734), 'rouge1_precision': tensor(0.2485), 'rouge1_recall': tensor(0.3837), 'rouge2_fmeasure': tensor(0.1123), 'rouge2_precision': tensor(0.1006), 'rouge2_recall': tensor(0.1646), 'rougeL_fmeasure': tensor(0.1828), 'rougeL_precision': tensor(0.1656), 'rougeL_recall': tensor(0.2617), 'rougeLsum_fmeasure': tensor(0.2517), 'rougeLsum_precision': tensor(0.2292), 'rougeLsum_recall': tensor(0.3534)}

# summn_large
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/summn/test.large.hypo", reference_file="./output/results/summn/test.target").calculate()
# BERTScore F1 score: 0.7471389174461365
# ROUGE scores: {'rouge1_fmeasure': tensor(0.3354), 'rouge1_precision': tensor(0.3697), 'rouge1_recall': tensor(0.3739), 'rouge2_fmeasure': tensor(0.1386), 'rouge2_precision': tensor(0.1507), 'rouge2_recall': tensor(0.1587), 'rougeL_fmeasure': tensor(0.2363), 'rougeL_precision': tensor(0.2600), 'rougeL_recall': tensor(0.2644), 'rougeLsum_fmeasure': tensor(0.3098), 'rougeLsum_precision': tensor(0.3419), 'rougeLsum_recall': tensor(0.3458)}

# bertscore_base
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/paragraph.bert/test.base.hypo", reference_file="./output/results/paragraph.bert/test.target").calculate()
# BERTScore F1 score: 0.6931549906730652
# ROUGE scores: {'rouge1_fmeasure': tensor(0.2377), 'rouge1_precision': tensor(0.2224), 'rouge1_recall': tensor(0.3241), 'rouge2_fmeasure': tensor(0.0891), 'rouge2_precision': tensor(0.0818), 'rouge2_recall': tensor(0.1286), 'rougeL_fmeasure': tensor(0.1568), 'rougeL_precision': tensor(0.1462), 'rougeL_recall': tensor(0.2186), 'rougeLsum_fmeasure': tensor(0.2169), 'rougeLsum_precision': tensor(0.2034), 'rougeLsum_recall': tensor(0.2953)}

# bertscore_large
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/paragraph.bert/test.large.hypo", reference_file="./output/results/paragraph.bert/test.target").calculate()
# BERTScore F1 score: 0.7407516241073608
# ROUGE scores: {'rouge1_fmeasure': tensor(0.3164), 'rouge1_precision': tensor(0.3527), 'rouge1_recall': tensor(0.3462), 'rouge2_fmeasure': tensor(0.1224), 'rouge2_precision': tensor(0.1322), 'rouge2_recall': tensor(0.1396), 'rougeL_fmeasure': tensor(0.2221), 'rougeL_precision': tensor(0.2454), 'rougeL_recall': tensor(0.2464), 'rougeLsum_fmeasure': tensor(0.2914), 'rougeLsum_precision': tensor(0.3253), 'rougeLsum_recall': tensor(0.3191)}

# rougescore_base
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/paragraph.rouge/test.base.hypo", reference_file="./output/results/paragraph.rouge/test.target").calculate()
# BERTScore F1 score: 0.6896432638168335
# ROUGE scores: {'rouge1_fmeasure': tensor(0.2277), 'rouge1_precision': tensor(0.2114), 'rouge1_recall': tensor(0.3137), 'rouge2_fmeasure': tensor(0.0816), 'rouge2_precision': tensor(0.0743), 'rouge2_recall': tensor(0.1182), 'rougeL_fmeasure': tensor(0.1468), 'rougeL_precision': tensor(0.1355), 'rougeL_recall': tensor(0.2068), 'rougeLsum_fmeasure': tensor(0.2083), 'rougeLsum_precision': tensor(0.1939), 'rougeLsum_recall': tensor(0.2867)}

# rougescore_large
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/paragraph.rouge/test.large.hypo", reference_file="./output/results/paragraph.rouge/test.target").calculate()
# BERTScore F1 score: 0.7388600707054138
# ROUGE scores: {'rouge1_fmeasure': tensor(0.3090), 'rouge1_precision': tensor(0.3488), 'rouge1_recall': tensor(0.3287), 'rouge2_fmeasure': tensor(0.1169), 'rouge2_precision': tensor(0.1283), 'rouge2_recall': tensor(0.1290), 'rougeL_fmeasure': tensor(0.2165), 'rougeL_precision': tensor(0.2428), 'rougeL_recall': tensor(0.2334), 'rougeLsum_fmeasure': tensor(0.2844), 'rougeLsum_precision': tensor(0.3214), 'rougeLsum_recall': tensor(0.3026)}

# paragraph.level_base
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/paragraph.level/test.base.hypo", reference_file="./output/results/paragraph.level/test.target").calculate()
# BERTScore F1 score: 0.6896432638168335
# ROUGE scores: {'rouge1_fmeasure': tensor(0.2277), 'rouge1_precision': tensor(0.2114), 'rouge1_recall': tensor(0.3137), 'rouge2_fmeasure': tensor(0.0816), 'rouge2_precision': tensor(0.0743), 'rouge2_recall': tensor(0.1182), 'rougeL_fmeasure': tensor(0.1468), 'rougeL_precision': tensor(0.1355), 'rougeL_recall': tensor(0.2068), 'rougeLsum_fmeasure': tensor(0.2083), 'rougeLsum_precision': tensor(0.1939), 'rougeLsum_recall': tensor(0.2867)}

# paragraph.level_large
# bert_score, rouge_score = EvaluateResults(hypothesis_file="./output/results/paragraph.level/test.large.hypo", reference_file="./output/results/paragraph.level/test.target").calculate()
# BERTScore F1 score: 0.7431467771530151
# ROUGE scores: {'rouge1_fmeasure': tensor(0.3237), 'rouge1_precision': tensor(0.3538), 'rouge1_recall': tensor(0.3622), 'rouge2_fmeasure': tensor(0.1268), 'rouge2_precision': tensor(0.1358), 'rouge2_recall': tensor(0.1466), 'rougeL_fmeasure': tensor(0.2262), 'rougeL_precision': tensor(0.2459), 'rougeL_recall': tensor(0.2550), 'rougeLsum_fmeasure': tensor(0.2993), 'rougeLsum_precision': tensor(0.3274), 'rougeLsum_recall': tensor(0.3353)}