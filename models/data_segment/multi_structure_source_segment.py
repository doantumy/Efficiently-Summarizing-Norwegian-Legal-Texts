import json
import os
from typing import Dict, List, Tuple
from utils.dataset import *
from loguru import logger
from models.data_segment.segmentor_core import SourceParagraphSplitter
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


class MultiStructureSourceSegmentor(object):
    def __init__(self, cfg, data: Dict[str, List[Dict[str, str]]] = None, labels: Dict[str, List] = None,
                 load_from_file=False):
        if isinstance(cfg, configparser.ConfigParser):
            self.input_max_token = int(cfg.get("CONFIG", "input_max_token"))
            self.data = None
            self.labels = None
        else:
            logger.info("Preparing MultiStructureSourceSegmentor")
            self.cur_stage = cfg.cur_stage  # add when running
            self.config = getattr(cfg, f"stage{self.cur_stage}")
            self.output_path = cfg.train.output_path # defined in training_args.py
            self.input_max_token = self.config.input_max_token

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
                # Ignore the labels input because the label is included in the data
                self.labels = labels
        self.splitter = SourceParagraphSplitter(self.input_max_token - 20)  # 20 is the number of tokens for prompt
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
    
    def segment(self, prompt: str = None) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, List[int]]]:
        """Segment source data into subgroups.

        Args:
            prompt (str, optional): Prompt for the task. Defaults to None.

        Returns:
            Tuple[Dict[str, list], Dict[str, list], Dict[str, List[int]]]: Input data (source and target)
        """
        logger.info("Running MultiStructureSourceSegmentor on data")
        for split, _ in self.data.items():
            # source segmentation, target duplication, record the counting file
            for i, (source, target) in enumerate(zip(self.data[split], self.labels[split])):
                # Each split in the data contains a list of data (Dict or List)
                # Group each sentence into subgroup
                # logger.info(f"source 1: {source}")
                
                split_src = self.splitter.segment_one_sample(source)
                # logger.info(f"split_src 1 {split_src}")
                if prompt is not None:  # Add prompt into the input, for example: "oppsummer: "
                    split_src = [prompt + item for item in split_src]

                # logger.info(f"split_src 2 {split_src}")
                # Each subgroup wil have the same target (summary)
                for src in split_src:
                    # logger.info(f"source 2: {src}")
                    self.split_source[split].append(src.strip() + '\n')
                    self.duplicated_target[split].append(target.strip() + '\n')

                self.count[split].append(len(split_src))

        return self.split_source, self.duplicated_target, self.count


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
