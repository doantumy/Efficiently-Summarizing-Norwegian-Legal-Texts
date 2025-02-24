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

class DataPrunner:
    def __init__(self, model_path=None, 
                 model_name="NbAiLab/nb-bert-base", 
                 threshold=None, use_local_threshold=False,
                 seeds=123, context_length=512, 
                 proportion=None, reformat=False, 
                 keep_order=True, batch_size=16,
                 file_name=None, splits=None):
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

        # Determine if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        os.system('nvidia-smi')

        # Load tokenizer and model
        if self.model_path:
            logger.info(f"Loading model from local folder: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels=1).to(self.device)
        else:
            logger.info(f"Loading model from Hugging Face: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1).to(self.device)

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

    def sentenization(self, text, length_limit=50, spacy_model="nb_core_news_sm", max_length=3_000_000):
        """
        Performs sentence segmentation using SpaCy.

        Args:
            text (str): Document text.
            length_limit (int): Minimum number of characters per sentence.
            spacy_model (str): SpaCy model to use for sentence segmentation.
            max_length (int): Maximum length for SpaCy NLP model.
        Returns:
            List[str]: A list of sentences.
        """
        if not spacy_model:
            raise ValueError("SpaCy NLP model is required for sentenization.")

        nlp = spacy.load(spacy_model)
        nlp.max_length=max_length
        doc = nlp(text)
        return [sent.text for sent in doc.sents if len(sent.text) >= length_limit]

    def save_to_pkl(self, split_name, batch_data, output_dir="data/all_processed_data/"):
        """
        Saves the processed data for a specific split to a .pkl file incrementally.

        Args:
            split_name (str): Name of the split being saved (e.g., 'train', 'val', 'test').
            batch_data (list): Processed data for the current batch.
            output_dir (str): Directory where the output file will be saved.

        Returns:
            None
        """
        file_path = f"{output_dir}{split_name}.pkl"

        if not os.path.exists(file_path):
            # Initialize a new file for the split
            with open(file_path, "wb") as f:
                pickle.dump([], f)

        # Append the new batch to the .pkl file
        with open(file_path, "rb") as f:
            existing_data = pickle.load(f)

        # Combine the existing data with the new batch
        existing_data.extend(batch_data)

        # Save the updated data back to the .pkl file
        with open(file_path, "wb") as f:
            pickle.dump(existing_data, f)

        logger.info(f"Saved batch data for split '{split_name}' to {file_path}")


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

    def shuffle(self, sentences=None, scores=None):
        random.seed(self.seeds)
        # Shuffle the sentences and their corresponding scores
        combined = list(zip(sentences, scores))
        random.shuffle(combined)  # Shuffle the combined list
        sentences, scores = zip(*combined)  # Unpack back to separate lists
        return list(sentences), list(scores)

    def filter_sentences(self, split_data, split_name=None, length_limit=50, log_errors=True, error_dir=None):
        """
        Filters sentences based on the threshold or proportion, with an option to maintain the original order.
        Short sentences below `length_limit` are removed. Documents with empty filtered_sentences are optionally logged to `error.json`.

        Args:
            split_data (list of dict): Data split to process, containing summaries and texts.
            split_name (str): Name of the data split (e.g., 'train', 'val', 'test'). None for text input.
            length_limit (int): Minimum number of characters a sentence must have to be included.
            log_errors (bool): Whether to log documents with empty filtered sentences.
            error_dir (str): Directory to log documents with empty filtered sentences.

        Returns:
            list of dict: Processed data with filtered sentences. Data points with empty sentences are optionally logged.
        """
        filtered_data = []
        if not self.proportion and not self.threshold and self.use_local_threshold:
            errors = {
                "config": {
                    "pruning-type": "proportion" if self.proportion else "threshold",
                    "pruning-value": "local threshold (avg of sim score per document)",
                    "reformat": self.reformat,
                    "keep-order": self.keep_order
                },
                split_name: [] if split_name else []
            }
        else:
            errors = {
                "config": {
                    "pruning-type": "proportion" if self.proportion else "threshold",
                    "pruning-value": self.proportion if self.proportion else self.threshold,
                    "reformat": self.reformat,
                    "keep-order": self.keep_order
                },
                split_name: [] if split_name else []
            }

        for item in split_data:
            # Remove short sentences before scoring
            sentences = [sentence for sentence in (item["text"] or []) if len(sentence) >= length_limit]

            if not sentences:
                if log_errors and split_name:
                    errors[split_name].append({
                        "reference": item.get("filename", ""),
                        "score": [],
                        "text": item["text"]
                    })
                continue  # Skip this data point if no sentences remain after filtering

            scores = self.predict_scores(sentences)

            if self.proportion:
                # Keep top N sentences based on the proportion
                n_keep = max(1, int(len(sentences) * self.proportion))
                top_sentences = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)[:n_keep]
                if self.keep_order:
                    top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[1]))  # Restore original order
                filtered_sentences = [sentence for _, sentence in top_sentences]
                filtered_scores = [score for score, _ in top_sentences]

                if not self.keep_order: # shuffle data
                    filtered_sentences, filtered_scores = self.shuffle(sentences=filtered_sentences, scores=filtered_scores)
            else:
                if self.use_local_threshold:
                    # Calculate local threshold per document
                    local_threshold = sum(scores) / len(scores)
                    filtered_sentences = [
                        sentence for sentence, score in zip(sentences, scores) if score >= local_threshold
                    ]
                    filtered_scores = [
                        score for _, score in zip(sentences, scores) if score >= local_threshold
                    ]
                    if self.keep_order:
                        filtered_sentences = [sentence for sentence in sentences if sentence in filtered_sentences]
                        filtered_scores = [
                            score for sentence, score in zip(sentences, scores) if sentence in filtered_sentences
                        ]
                    else: # Shuffle data
                        if len(filtered_sentences) and len(filtered_scores) > 1:
                            filtered_sentences, filtered_scores = self.shuffle(sentences=filtered_sentences, scores=filtered_scores)
                else:
                    # Filter based on global threshold
                    filtered_sentences = [
                        sentence for sentence, score in zip(sentences, scores) if score > self.threshold
                    ]
                    filtered_scores = [
                        score for _, score in zip(sentences, scores) if score > self.threshold
                    ]
                    if not self.keep_order: # Shuffle data
                        if len(filtered_sentences) and len(filtered_scores) > 1:
                            filtered_sentences, filtered_scores = self.shuffle(sentences=filtered_sentences, scores=filtered_scores)

            if filtered_sentences:
                # Add the data point if filtered_sentences is not empty
                filtered_data.append({
                    "summary": item.get("summary", ""),
                    "reference": item.get("filename", ""),
                    "score": filtered_scores,
                    "text": filtered_sentences
                })
            else:
                if log_errors and split_name:
                    # Log the error for empty filtered_sentences
                    errors[split_name].append({
                        "reference": item.get("filename", ""),
                        "score": scores,
                        "text": sentences
                    })

        # Append errors to the error file if applicable
        suffix = (
            f"errors_threshold-{self.threshold}" if self.proportion is None else f"_proportion-{int(self.proportion * 100)}"
        )
        suffix += f"_reformat-{self.reformat}_order-{self.keep_order}"
        if log_errors and split_name and errors[split_name]:
            self.log_errors_to_file(errors=errors, error_file=f"{error_dir}{suffix}.json")

        return filtered_data

    def log_errors_to_file(self, errors, error_file):
        """
        Appends errors to the specified error file.

        Args:
            errors (dict): Errors to log, keyed by split name.
            error_file (str): Path to the error log file.

        Returns:
            None
        """
        try:
            # Read existing errors from the file if it exists
            with open(error_file, 'r', encoding='utf-8') as file:
                existing_errors = json.load(file)
        except FileNotFoundError:
            existing_errors = {}

        # Update the existing errors with new errors
        existing_errors.update(errors)

        # Write back the updated errors to the file
        with open(error_file, 'w', encoding='utf-8') as file:
            json.dump(existing_errors, file, indent=4, ensure_ascii=False)


    def preprocess(self):
        """
        Preprocesses the loaded data (JSON or text) by streaming and processing each split in batches.
        Saves each split incrementally as .pkl files and combines them into a single JSON file at the end.

        Returns:
            None
        """
        for split in self.splits:
            logger.info(f"Processing split: {split}")
            idx = 1
            # Stream and process data in batches
            for batch in self.load_data_stream(split):
                logger.info(f"Process batch id: {idx}")
                # Filter sentences and log errors for each batch
                processed_batch = self.filter_sentences(
                    split_data=batch,
                    split_name=split,
                    log_errors=True,
                    error_dir="data/all_processed_data/"
                )

                # Optionally reformat the processed batch
                if self.reformat:
                    processed_batch = self.reformat_text_to_chapter_structure(processed_batch)

                # Save the processed batch incrementally as .pkl
                self.save_to_pkl(split_name=split, batch_data=processed_batch)
                idx += 1  # Increment the batch index for the next batch

            logger.info(f"Completed processing split: {split}")

        # Combine all splits into a single JSON file after all splits are processed
        self.combine_splits_to_json(output_dir="data/all_processed_data/")
        logger.info("All splits have been combined into a single JSON file.")

    def combine_splits_to_json(self, output_dir="data/all_processed_data/"):
        """
        Combines all splits (train, val, test) stored as .pkl files into a single JSON file.

        Args:
            output_dir (str): Directory where the .pkl files are stored.

        Returns:
            None
        """
        combined_data = {"train": [], "val": [], "test": []}  # Initialize the structure for all splits

        for split in self.splits:
            file_path = f"{output_dir}{split}.pkl"

            if os.path.exists(file_path):
                # Load the .pkl file for the split
                with open(file_path, "rb") as f:
                    split_data = pickle.load(f)
                    combined_data[split].extend(split_data)  # Ensure the split data is added to the correct key
                # Delete the .pkl file after loading
                os.remove(file_path)
                logger.info(f"Deleted temporary file: {file_path}")
            else:
                logger.warning(f"No data found for split '{split}'")

        # Generate the file name for the combined JSON
        pruning_type = "proportion" if self.proportion else "threshold"
        if self.proportion:
            pruning_value = self.proportion
        elif self.threshold:
            pruning_value = self.threshold
        elif self.use_local_threshold:
            pruning_value = "local"
        # pruning_value = self.proportion if self.proportion else self.threshold
        reformat = "reformat_true" if self.reformat else "reformat_false"
        order = "order_true" if self.keep_order else "order_false"

        file_name = f"all_splits_{pruning_type}_{pruning_value}_{reformat}_{order}.json"
        full_filepath = f"{output_dir}{file_name}"

        # Save the combined data to a JSON file
        with open(full_filepath, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)

        logger.info(f"Combined data saved to {full_filepath}")

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
            sentences = item["text"]

            # Create a nested chapter structure
            chapter_structure = {
                "KAPITTEL_0": {
                    "KAPITTEL_0-0": sentences  # Subchapter with the list of sentences
                }
            }

            # Append reformatted data
            reformatted_data.append({
                "summary": item["summary"],
                "reference": item.get("filename", ""),
                "score": item.get("score", []),  # Include scores at the top level if needed
                "text": chapter_structure
            })

        return reformatted_data
    
    
if __name__ == "__main__":
    # # input_file = "data/processed_data/test.sent.json"
    # # do_shuffle = [False, True]
    do_reformat = [False, True]
    keep_order = [True, False]
    proportion_values = [0.3, 0.5, 0.7]
    batch_size = 64
    splits = ["train", "val", "test"]
    
    # # Norwegian WEB data
    # # input_file = "data/processed_data/processed_data_wo_segment_w_sentenization.json"
    # # threshold_values = [0.58, 0.65, 0.72]

    # Norwegian ALL LOVdata
    input_file = "data/all_processed_data/all_data_wo_segment_w_sent.json"
    classifier_path = "output/regression.cls.20241214_173636/trainer_output/checkpoint-18414"
    threshold_values = [0.57, 0.64, 0.71]
    
    # # Australian data
    # input_file = "data/au_data_processed/au_processed_data_wo_segment_w_sentenization.json"
    # classifier_path = "output/regression.cls.20241130_185749/trainer_output/checkpoint-8841"
    # threshold_values = [0.58, 0.68, 0.78]

    for order in keep_order:
        for reformat in do_reformat:
            for proportion in proportion_values:
                if reformat:
                    logger.info(f"ORDER: {order} Para method. Proportion {proportion}. Reformat {reformat}")
                else:
                    logger.info(f"ORDER: {order} SUMM^N method. Proportion {proportion}. Reformat {reformat}")

                pruner = DataPrunner(
                    model_path=classifier_path,
                    threshold=None, use_local_threshold=False,
                    proportion=proportion,
                    reformat=reformat,
                    keep_order=order,
                    batch_size=batch_size, 
                    file_name=input_file, splits=splits
                )
                pruner.preprocess()
            
            # Local threshold
            logger.info(f"ORDER: {order} SUMM^N method. Threshold LOCAL. Reformat {reformat}")
            pruner = DataPrunner(
                model_path=classifier_path,
                threshold=None, use_local_threshold=True,
                proportion=None,
                reformat=reformat,
                keep_order=order,
                batch_size=batch_size, 
                file_name=input_file, splits=splits
            )
            pruner.preprocess()
            
            for threshold in threshold_values:
                if reformat:
                    logger.info(f"ORDER: {order} Para method. Threshold {threshold}. Reformat {reformat}")
                else:
                    logger.info(f"ORDER: {order} SUMM^N method. Threshold {threshold}. Reformat {reformat}")

                pruner = DataPrunner(
                    model_path=classifier_path,
                    threshold=threshold, use_local_threshold=False,
                    proportion=None,
                    reformat=reformat,
                    keep_order=order,
                    batch_size=batch_size, 
                    file_name=input_file, splits=splits
                )
                pruner.preprocess()

    
    
    




    # TEST
    # Shuffle data based on proportion, set threshold=None, use_local_threshold=False
    # pruner = DataPrunner(
    #     model_path="output/regression.cls.20241120_125844/trainer_output/checkpoint-35",
    #     threshold=None, use_local_threshold=False,
    #     proportion=0.8,
    #     reformat=False,
    #     keep_order=False,
    #     batch_size=16, file_name=input_file, splits=["train", "val", "test"]
    # )
    # pruner.preprocess()

    # # Shuffle data based on threshold, set either threshold=0.x or use_local_threshold=True
    # pruner = DataPrunner(
    #     model_path="output/regression.cls.20241120_125844/trainer_output/checkpoint-35",
    #     threshold=None, use_local_threshold=True,
    #     proportion=None,
    #     reformat=True,
    #     keep_order=False,
    #     batch_size=16, file_name=input_file, splits=["train", "val", "test"]
    # )
    # pruner.preprocess()
    
    # # Shuffle - local threshold
    # pruner = DataPrunner(
    #     model_path="output/regression.cls.20241120_125844/trainer_output/checkpoint-35",
    #     threshold=None, use_local_threshold=True,
    #     proportion=None,
    #     reformat=False,
    #     keep_order=True,
    #     batch_size=16, file_name=input_file, splits=["train", "val", "test"]
    # )
    # pruner.preprocess()
    