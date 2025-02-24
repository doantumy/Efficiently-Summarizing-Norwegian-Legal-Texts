import hashlib
import json
import multiprocessing
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Union
from models.data_segment.segmentor_core import SourceSimilarityScoreSplitter
import os
import pickle as pkl
from utils.dataset import *
from tqdm import tqdm
from loguru import logger
import configparser

"""
Example data:
train_samples = [
    {"id": 1, "text": "This is a training sample", "label": "A"},
    {"id": 2, "text": "This is another training sample", "label": "B"},
]

val_samples = [
    {"id": 3, "text": "This is a validation sample", "label": "A"},
    {"id": 4, "text": "This is another validation sample", "label": "B"},
]

test_samples = [
    {"id": 5, "text": "This is a test sample", "label": "A"},
    {"id": 6, "text": "This is another test sample", "label": "B"},
]

data_splits = {
    "train": train_samples,
    "val": val_samples,
    "test": test_samples
}
"""


class SourceSimilaritySegmentor(object):
    """Handle segmentation of data based on similarity scores.

    Args:
        object (any): input object
    """

    def __init__(self, cfg, data: Dict[str, List[Dict[str, str]]] = None, labels: Dict[str, List] = None,
                 load_from_file=False, metric: str = "bertscore"):
        logger.info(f"Preparing SourceSimilaritySegmentor with metric: {metric}")
        if isinstance(cfg, configparser.ConfigParser):
            self.input_max_token = int(cfg.get("CONFIG", "input_max_token"))
            self.data = None
            self.labels = None
            self.cache_path = "/tmp"
        else:
            self.cur_stage = cfg.cur_stage  # add when running
            self.config = getattr(cfg, f"stage{self.cur_stage}")
            self.output_path = cfg.train.output_path  # defined in training_args.py
            self.cache_path = self.output_path.replace(self.output_path.split('.')[-1], "cache")
            self.reuse = getattr(cfg.run_args, f"reuse_stage{self.cur_stage}", "False") == "True"
            self.similarity_metric = metric
            self.nthreads = min(
                [multiprocessing.cpu_count(), self.config.cores_used, 1])  # force no multiprocessing here
            self.input_max_token = cfg.run_args.input_max_token
            if load_from_file:
                # Load source and target data
                for split, _ in data.items():
                    # Use the coarse summary as the input for the next stage, except at stage 0
                    if self.cur_stage == 1:
                        suffix = 'source'
                    else:
                        suffix = 'hypo'
                    # Read data from the output of previous stage
                    stage_path = os.path.join(self.output_path, f"stage_{self.cur_stage - 1}")
                    data_path = os.path.join(stage_path, f"{split}.{suffix}")
                    self.data = read_list_asline(data_path)
                    label_path = os.path.join(self.output_path, "stage_0", f"{split}.target")
                    self.labels = read_list_asline(label_path)
            else:
                self.data = data
                self.labels = labels
        self.splitter = SourceSimilarityScoreSplitter(metric=metric,
                                                      input_max_token=self.input_max_token - 20)  # 20 is the number of tokens for prompt
        # store segmented source and targets
        self.split_source = {'train': [], 'val': [], 'test': []}
        self.duplicated_target = {'train': [], 'val': [], 'test': []}
        self.count = {'train': [], 'val': [], 'test': []}

    def execute(self, data: Dict[str, List[Dict[str, str]]], labels: Dict[str, List[str]]):
        self.data = data
        self.labels = labels
        self.split_source = {'train': [], 'val': [], 'test': []}
        self.duplicated_target = {'train': [], 'val': [], 'test': []}
        self.count = {'train': [], 'val': [], 'test': []}

        return self.segment(prompt="oppsummer: ")

    def _calculate_similarity_and_split(self, i: int, source: dict, target: str, prompt: str = None):
        # Each split in the data contains a list of data (Dict or List)
        # Group each sentence into subgroup
        split_src = self.splitter.similar_sentences_segment(source)

        if prompt is not None:  # Add prompt into the input, for example: "oppsummer: "
            split_src = [prompt[i]] + split_src

        split_source = []
        duplicated_target = []
        # Each subgroup wil have the same target (summary)
        for src in split_src:
            split_source.append(src.strip() + '\n')
            duplicated_target.append(target.strip() + '\n')

        count = len(split_src)
        return i, split_source, duplicated_target, count

    def segment(self, prompt: str = None) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, List[int]]]:
        """Segment source data into subgroups.

        Args:
            prompt (str, optional): Prompt for the task. Defaults to None.

        Returns:
            Tuple[Dict[str, list], Dict[str, list], Dict[str, List[int]]]: source and target data
        """
        if self.reuse:
            logger.info("REUSE: Checking preprocessed data")
            split_source, duplicated_target, count = self._load_data()
            if split_source is not None:
                self.split_source = split_source
                self.duplicated_target = duplicated_target
                self.count = count
                return self.split_source, self.duplicated_target, self.count

        logger.info(f"Running SourceSimilaritySegmentor on data with metric {self.similarity_metric}")
        for split, _ in self.data.items():
            # source segmentation, target duplication, record the counting file
            iter_segments = enumerate(zip(self.data[split], self.labels[split]))
            results = [self._calculate_similarity_and_split(i, source, target, prompt)
                       for i, (source, target) in tqdm(iter_segments, desc=f"Segmenting {split}",
                                                       total=len(self.data[split]))]
            # sort results by the index
            results = sorted(results, key=lambda x: x[0])
            for i, split_src, split_tgt, count in results:
                self.split_source[split] += split_src
                self.duplicated_target[split] += split_tgt
                self.count[split].append(count)

        # dump data after processing to save time in case of failure
        self._dump_data(self.split_source, self.duplicated_target, self.count)

        return self.split_source, self.duplicated_target, self.count

    def _dump_data(self, split: dict, target: dict, count: dict):
        # use a global cache folder outside the experiment to store the preprocessed data
        stage_path = os.path.join(self.cache_path, f"stage_{self.cur_stage}")

        # create pkl file path and check that parent exists
        pkl_file_path = os.path.join(stage_path, f"{self.similarity_metric}_source_similarity_segmented_data.pkl")
        Path(pkl_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save the segmented source and targets to disk
        full = {"source": split, "target": target, "count": count}
        with open(pkl_file_path, 'wb') as file:
            pkl.dump(full, file, protocol=pkl.HIGHEST_PROTOCOL)

        # Save the hash of the original data to check the integrity of the data for reuse
        serial_original_data = pkl.dumps(self.data)
        logger.info("Creating hash to check the integrity of source the data for reuse")
        md_hash = hashlib.md5(serial_original_data).hexdigest()
        with open(os.path.join(stage_path, f"{self.similarity_metric}_source_similarity_segmented_data_hash.txt"),
                  'w') as file:
            file.write(md_hash)

    def _load_data(self):
        # use a global cache folder outside the experiment to store the preprocessed data
        stage_path = os.path.join(self.cache_path, f"stage_{self.cur_stage}")

        # Construct the full file path
        file_path = os.path.join(stage_path, f"{self.similarity_metric}_source_similarity_segmented_data.pkl")

        if not os.path.exists(file_path):
            return None, None, None

        # Check if the hash is equal to the new source data
        logger.info("Checking hash to verify the integrity of the data for reuse")
        serial_original_data = pkl.dumps(self.data)
        md_hash = hashlib.md5(serial_original_data).hexdigest()
        with open(os.path.join(stage_path, f"{self.similarity_metric}_source_similarity_segmented_data_hash.txt"),
                  'r') as file:
            if md_hash != file.read():
                logger.error("Hash does not match, the data is corrupted, need to be recalculated (REUSE: FAILED)")
                return None, None, None

        logger.info("Hash matched, loading preprocessed data (REUSE: OK)")
        with open(os.path.join(stage_path, f"{self.similarity_metric}_source_similarity_segmented_data.pkl"),
                  'rb') as file:
            full = pkl.load(file)

        if len(full['source']['train']) < len(self.data['train']):
            logger.info("Data is not up-to-date, need to be recalculated (REUSE: FAILED)")
            return None, None, None
        return full['source'], full['target'], full['count']

    def save(self):
        """
        Save the segmented source and targets
        """
        stage_path = os.path.join(self.output_path, f"stage_{self.cur_stage}")
        if not os.path.exists(stage_path):
            os.makedirs(stage_path)

        for split, _ in self.data.items():
            source_output_path = os.path.join(stage_path, f"{split}.source")
            target_output_path = os.path.join(stage_path, f"{split}_duplicated.target")
            count_output_path = os.path.join(stage_path, f"{split}_count.json")

            write_list_asline(source_output_path, self.split_source[split])
            write_list_asline(target_output_path, self.duplicated_target[split])

            with open(count_output_path, 'w', encoding='utf-8') as file:
                json.dump(self.count[split], file, ensure_ascii=False, indent=4)
