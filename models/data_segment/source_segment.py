import json
import os
from typing import Dict, List, Tuple

from tqdm import tqdm

from utils.dataset import *
from loguru import logger
from models.data_segment.segmentor_core import SourceSplitterCore
import configparser

class SourceSegmentor(object):
    def __init__(self, cfg, data: Dict[str, List] = None, labels: Dict[str, List] = None, load_from_file=True):
        logger.info("Preparing SourceSegmentor")
        
        if isinstance(cfg, configparser.ConfigParser):
            self.input_max_token = int(cfg.get("CONFIG", "input_max_token"))
            self.data = None
            self.labels = None
        else:
            self.cur_stage = cfg.cur_stage
            self.cfg = getattr(cfg, f"stage{self.cur_stage}")
            self.output_path = cfg.train.output_path
            self.input_max_token = self.cfg.input_max_token

            # load source and targets
            if load_from_file is True:
                for data_type in ['train', 'val', 'test']:
                    # we use the coarse summary as the intput to the next stage, except for stage 0
                    if self.cur_stage == 1:
                        suffix = 'source'
                    else:
                        suffix = 'hypo'

                    # read data from the output of previous stage
                    stage_path = os.path.join(self.output_path, f"stage_{self.cur_stage - 1}")

                    data_path = os.path.join(stage_path, f"{data_type}.{suffix}")
                    self.data = read_list_asline(data_path)
                    # we always pick the target from stage 0 as the input target
                    label_path = os.path.join(self.output_path, "stage_0", f"{data_type}.target")
                    self.labels = read_list_asline(label_path)
            else:
                self.data = data
                self.labels = labels
        self.splitter = SourceSplitterCore(self.input_max_token - 34)  # 34 tokens for question, if any
        # store segmented source and targets
        self.split_source = {'train': [], 'val': [], 'test': []}
        self.dupli_target = {'train': [], 'val': [], 'test': []}
        self.count = {'train': [], 'val': [], 'test': []}
        
    def execute(self, data: Dict[str, List[Dict[str, str]]], labels: Dict[str, List[str]]):
        self.data = data
        self.labels = labels
        self.split_source = {'train': [], 'val': [], 'test': []}
        self.duplicated_target = {'train': [], 'val': [], 'test': []}
        self.count = {'train': [], 'val': [], 'test': []}

        return self.segment(prompt="oppsummer: ")

    def segment(self, query=None) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, List[int]]]:
        logger.info("Running SourceSegmentor on data")
        for data_type in ['train', 'val', 'test']:
            # source segmentation, target duplication, record the counting file
            for i, (trans, target) in tqdm(enumerate(zip(self.data[data_type], self.labels[data_type])),
                                           desc=f"Segmenting {data_type}"):
                # group each sentences into subgroups
                split_trans = self.splitter.segment_one_sample(trans)
                # make one string out of each subgroup
                split_trans = [' '.join(x) for x in split_trans]
                if query is not None:
                    split_trans = [query[i]] + split_trans

                # Each subgroup gets the full target (summary)
                for tran in split_trans:
                    self.split_source[data_type].append(tran.strip() + '\n')
                    self.dupli_target[data_type].append(target.strip() + '\n')

                self.count[data_type].append(len(split_trans))
        return self.split_source, self.dupli_target, self.count

    def save(self):
        stage_path = os.path.join(self.output_path, f"stage_{self.cur_stage}")
        if not os.path.exists(stage_path):
            os.makedirs(stage_path)

        for data_type in ['train', 'val', 'test']:
            source_output_path = os.path.join(stage_path, f"{data_type}.source")
            target_output_path = os.path.join(stage_path, f"{data_type}_duplicated.target")
            count_output_path = os.path.join(stage_path, f"{data_type}_count.json")

            write_list_asline(source_output_path, self.split_source[data_type])
            write_list_asline(target_output_path, self.dupli_target[data_type])

            with open(count_output_path, 'w', encoding='utf-8') as file:
                json.dump(self.count[data_type], file)
