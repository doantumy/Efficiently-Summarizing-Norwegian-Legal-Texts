import json
import os
import ijson
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
from itertools import islice
import pickle
import random

class DataPruner:
    def __init__(self, model_path=None, 
                 model_name="NbAiLab/nb-bert-base", 
                 threshold=None, use_local_threshold=False,
                 seeds=123, context_length=512, 
                 proportion=None, reformat=False, 
                 keep_order=True, batch_size=16,
                 file_name=None, splits=None, data_directory="data/all_processed_data"):
        """
        Initializes the DataPrunner class with batch processing and automatic file handling.

        Args:
            model_path (str): Path to a local pre-trained model directory. If None, the model is loaded from Hugging Face.
            model_name (str): Name of the pre-trained model on Hugging Face. Ignored if `model_path` is provided.
            context_length (int): Context length of tokenizer. Default is 512.
            threshold (float): Score threshold for filtering sentences.
            use_local_threshold (bool): ONLY WORK FOR THRESHOLD option.
                                        Whether to use the local threshold by averaging the sim score in the document. 
                                        If False, the threshold is determined from the training data.
            seeds (int): Random seed for reproducibility.
            proportion (float): Proportion of sentences to keep (0.0 to 1.0). If provided, overrides `threshold`.
            reformat (bool): Whether to reformat the text into a chapter structure (used for text input).
                             Set to False to format text for SUMM^N method. 
                             Set to True for paragraph.level and paragraph.BERT/ROUGE method.
            keep_order (bool): Whether to maintain the original order of sentences in the output.
            batch_size (int): Number of sentences to process in a single batch for inference.
        """
        self.model_name = model_name
        self.model_path = model_path
        self.threshold = threshold
        self.use_local_threshold = use_local_threshold
        self.seeds = seeds
        self.proportion = proportion
        self.reformat = reformat
        self.keep_order = keep_order
        self.batch_size = batch_size
        self.file_name = file_name
        self.splits = splits
        self.data_directory = data_directory

        # Determine if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        os.system('nvidia-smi')

        # Load tokenizer and model
        logger.info(f"Loading model from local folder: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=1).to(self.device)
        
            

        self.context_length = context_length

            
    def load_data_stream(self, split_name):
        """
        Load data from a specific split in a large JSON file using streaming.

        Args:
            file_name (str): Path to the JSON file.
            split_name (str): Name of the split to load (e.g., 'train', 'val', 'test').

        Yields:
            dict: Each JSON object (document) in the specified split.
        """
        with open(self.file_name, "r", encoding="utf-8") as f:
            # Stream data from the specified split
            items = ijson.items(f, f"{split_name}.item")
            
            # Batch the streamed items
            while True:
                batch = list(islice(items, self.batch_size))
                if not batch:
                    break
                yield batch

    def predict_scores(self, text):
        """
        Predicts scores for a list of sentences in batches using the model.

        Sentences are skipped if:
        - They are shorter than 50 characters.

        Args:
            text (list of str): List of sentences to score.

        Returns:
            list of float: Predicted scores for valid sentences.
        """
        valid_sentences = [sentence for sentence in text if len(sentence) >= 50]  # Filter short sentences

        if not valid_sentences:
            return []

        # Batch processing
        scores = []
        for i in range(0, len(valid_sentences), self.batch_size):
            batch_sentences = valid_sentences[i:i + self.batch_size]
            batch_tokens = self.tokenizer(batch_sentences, truncation=True, max_length=self.context_length, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch_tokens)
                batch_scores = outputs.logits.squeeze().tolist()  # regression task
                if isinstance(batch_scores, float):  # Single sentence batch edge case
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)

        return scores

    def load_data(self, split_name):
        file_path = os.path.join(self.data_directory, f"temp_{split_name}.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def shuffle(self, sentences=None, scores=None):
        random.seed(self.seeds)
        # Shuffle the sentences and their corresponding scores
        combined = list(zip(sentences, scores))
        random.shuffle(combined)  # Shuffle the combined list
        sentences, scores = zip(*combined)  # Unpack back to separate lists
        return list(sentences), list(scores)
    
    def process_and_filter_data(self, data):
        filtered_data = []

        for item in data:
            sentences = item['sentences']
            scores = item['scores']

            if self.proportion:
                n_keep = max(1, int(len(sentences) * self.proportion))
                top_sentences = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)[:n_keep]
                if self.keep_order:
                    top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[1]))
                filtered_sentences = [sentence for _, sentence in top_sentences]
                filtered_scores = [score for score, _ in top_sentences]

                if not self.keep_order:
                    filtered_sentences, filtered_scores = self.shuffle(sentences=filtered_sentences, scores=filtered_scores)
            else:
                if self.use_local_threshold:
                    local_threshold = sum(scores) / len(scores)
                    filtered_sentences = [sentence for sentence, score in zip(sentences, scores) if score >= local_threshold]
                    filtered_scores = [score for _, score in zip(sentences, scores) if score >= local_threshold]
                    if self.keep_order:
                        filtered_sentences = [sentence for sentence in sentences if sentence in filtered_sentences]
                        filtered_scores = [score for sentence, score in zip(sentences, scores) if sentence in filtered_sentences]
                    else:
                        if len(filtered_sentences) and len(filtered_scores) > 1:
                            filtered_sentences, filtered_scores = self.shuffle(sentences=filtered_sentences, scores=filtered_scores)
                else:
                    filtered_sentences = [sentence for sentence, score in zip(sentences, scores) if score > self.threshold]
                    filtered_scores = [score for _, score in zip(sentences, scores) if score > self.threshold]
                    if not self.keep_order:
                        if len(filtered_sentences) and len(filtered_scores) > 1:
                            filtered_sentences, filtered_scores = self.shuffle(sentences=filtered_sentences, scores=filtered_scores)

            if filtered_sentences:
                filtered_data.append({
                    "summary": item.get("summary", ""),
                    "filename": item.get("filename", ""),
                    "scores": filtered_scores,
                    "text": filtered_sentences
                })

        return filtered_data

    def all_scores_process(self):
        """
        Pre-calculated all scores for all splits
        Preprocesses the loaded data (JSON or text) by streaming and processing each split in batches.
        Saves each split incrementally as JSON

        Returns:
            None
        """
        for split in self.splits:
            file_path = os.path.join(self.data_directory, f"temp_{split}.json")
            first_write = True
            with open(file_path, 'w') as f:
                f.write('[')  # Start JSON array
                for idx, batch in enumerate(self.load_data_stream(split), 1):
                    logger.info(f"Processing split: {split}, batch id: {idx}")
                    for item in batch:
                        sentences = [sentence for sentence in (item["text"] or []) if len(sentence) >= 50]
                        scores = self.predict_scores(sentences)
                        ref_name = item["filename"]
                        summary = item["summary"]
                        data = {
                            "sentences": sentences,
                            "scores": scores,
                            "filename": ref_name,
                            "summary": summary
                        }
                        if not first_write:
                            f.write(',')  # JSON array element separator
                        json.dump(data, f, indent=4, ensure_ascii=False)
                        first_write = False
                f.write(']')  # End JSON array
                logger.info(f"Data for split {split} saved to {self.data_directory}")

        logger.info("All splits have been combined into a single JSON file.")

    
    def reformat_text_to_chapter_structure(self, split_data):
        """
        Reformat the text into a nested chapter structure.

        Args:
            split_data (list): List of data items containing "text" and "summary" fields.

        Returns:
            list: List of reformatted data items.
        """
        reformatted_data = []
        for item in split_data:
            # Create a nested chapter structure
            chapter_structure = {
                "KAPITTEL_0": {
                    "KAPITTEL_0-0": item["text"]  # Subchapter with the list of sentences
                }
            }

            # Append reformatted data
            reformatted_data.append({
                "summary": item["summary"],
                "filename": item["filename"],
                "scores": item["scores"],  # Include scores at the top level if needed
                "text": chapter_structure
            })

        return reformatted_data
    def process_splits(self):
        results = {}
        for split in self.splits:
            data = self.load_data(split_name=split)
            filtered_data = self.process_and_filter_data(data)
            if self.reformat:
                filtered_data = self.reformat_text_to_chapter_structure(filtered_data)

            results[split] = filtered_data

        # Save all results into one JSON file
        self.save_results(results)

    def save_results(self, results):
        # Generate the file name for the combined JSON
        pruning_type = "proportion" if self.proportion else "threshold"
        if self.proportion:
            pruning_value = self.proportion
        elif self.threshold:
            pruning_value = self.threshold
        elif self.use_local_threshold:
            pruning_value = "local"

        reformat = "reformat_true" if self.reformat else "reformat_false"
        order = "order_true" if self.keep_order else "order_false"

        file_name = f"all_splits_{pruning_type}_{pruning_value}_{reformat}_{order}.json"

        output_path = os.path.join(self.data_directory, file_name)
        with open(output_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        logger.info(f"All filtered results saved to {output_path}")


if __name__ == "__main__":
    # input_file = "data/all_processed_data/test.sent.json"
    # classifier_path = "output/regression.cls.20241120_125844/trainer_output/checkpoint-35"
    
    # Norwegian ALL LOVdata
    input_file = "data/all_processed_data/all_data_wo_segment_w_sent.json"
    classifier_path = "output/regression.cls.20241214_173636/trainer_output/checkpoint-18414"
    do_reformat = [False, True]
    keep_order = [True, False]

    proportion_values = [0.3, 0.5, 0.7]
    threshold_values = [0.57, 0.64, 0.71]
    batch_size = 64
    splits = ["train", "val", "test"]
    # DataPruner(file_name=input_file, splits=splits, model_path=classifier_path).all_scores_process()
    
    # data_processor = DataPruner(splits=splits, data_directory="data/all_processed_data", 
    #                             proportion=0.5, keep_order=True, use_local_threshold=False, 
    #                             threshold=None, reformat='basic')
    # data_processor.process_splits()


    # pruner = DataPruner(
    #                 splits=splits, data_directory="data/all_processed_data",
    #                 model_path=classifier_path,
    #                 threshold=None, use_local_threshold=True,
    #                 proportion=None,
    #                 reformat=True,
    #                 keep_order=True,
    #                 batch_size=batch_size, 
    #                 file_name=input_file
    #             )
    # pruner.process_splits()

    for order in keep_order:
        for reformat in do_reformat:
            # Proportion
            for proportion in proportion_values:
                if reformat:
                    logger.info(f"ORDER: {order} Para method. Proportion {proportion}. Reformat {reformat}")
                else:
                    logger.info(f"ORDER: {order} SUMM^N method. Proportion {proportion}. Reformat {reformat}")

                pruner = DataPruner(
                    splits=splits, data_directory="data/all_processed_data",
                    threshold=None, use_local_threshold=False,
                    model_path=classifier_path,
                    proportion=proportion,
                    reformat=reformat,
                    keep_order=order,
                    batch_size=batch_size, 
                    file_name=input_file
                )
                pruner.process_splits()
            
            # Local threshold
            logger.info(f"ORDER: {order} SUMM^N method. Threshold LOCAL. Reformat {reformat}")
            pruner = DataPruner(
                splits=splits, data_directory="data/all_processed_data",
                threshold=None, use_local_threshold=True,
                model_path=classifier_path,
                proportion=None,
                reformat=reformat,
                keep_order=order,
                batch_size=batch_size, 
                file_name=input_file
            )
            pruner.process_splits()
            
            # Threshold
            for threshold in threshold_values:
                if reformat:
                    logger.info(f"ORDER: {order} Para method. Threshold {threshold}. Reformat {reformat}")
                else:
                    logger.info(f"ORDER: {order} SUMM^N method. Threshold {threshold}. Reformat {reformat}")

                pruner = DataPruner(
                splits=splits, data_directory="data/all_processed_data",
                    threshold=threshold, use_local_threshold=False,
                    model_path=classifier_path,
                    proportion=None,
                    reformat=reformat,
                    keep_order=order,
                    batch_size=batch_size, 
                    file_name=input_file
                )
                pruner.process_splits()