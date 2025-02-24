import gzip
import os
import json
from loguru import logger

from utils.dataset import *
from dataset_loader.base_loader import LoaderBase


class Loader(LoaderBase):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load(self):

        data_path = self.cfg.train.dataset_path
        with open(data_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        """
        # data is a dictionary with the following structure:
        {
        "train": [{"summary": "summary text", "reference": "reference text", "text": "text"}, …],
        "val": [{"summary": "summary text", "reference": "reference text", "text": "text"}, …],
        "test": [{"summary": "summary text", "reference": "reference text", "text": "text"}, …]
        }
        """

        for data_type in ['train', 'val', 'test']:
            samples = data[data_type]
            if self.cfg.run_args.debug:
                logger.info("Debug mode is ON. Using only 5% of the data.")
                samples = samples[:len(samples) // 50]
            for sample in samples:
                # get text & summary
                text = sample["text"]
                summary = sample["summary"]

                if len(text) == 0 or len(summary) == 0:
                    continue

                self.data[data_type].append(text)
                self.label[data_type].append(summary)

        return self.data, self.label
