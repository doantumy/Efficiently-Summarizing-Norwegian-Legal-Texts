from typing import Dict

from torch.cuda import empty_cache
from tqdm import tqdm
import os
from loguru import logger

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from utils.dataset import write_list_asline
from utils.gpu import free_memory


class SummaryGenerator(object):
    def __init__(self, cfg, data: Dict[str, list], fine_grained=False, test_mode=False):
        self.cfg = cfg
        self.stage_cfg = getattr(cfg, f"stage{cfg.cur_stage}")
        self.data = data
        self.fine_grained = fine_grained
        self.test_mode = test_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if os.getenv('DEBUG', False) == 'True':
            logger.debug("DEBUG MODE: skipping | force everything to CPU")
            self.device = "cpu"

        logger.info(f"Loading model from {self.stage_cfg.trainer_output_folder}")

        if cfg.run_args.use_quantize:
            # Quantization as defined https://huggingface.co/docs/optimum/concept_guides/quantization will help us reduce the size of the model for it to fit on a single GPU
            # Quantization configuration
            compute_dtype = getattr(torch, "float16")
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.stage_cfg.trainer_output_folder,
                trust_remote_code=True,
                quantization_config=self.bnb_config,
                ignore_mismatched_sizes=True,
            ).to(self.device)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.stage_cfg.trainer_output_folder,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.stage_cfg.trainer_output_folder
        )

        # Store generated samples
        self.hypo = {'train': [], 'val': [], 'test': []}

    def run_summary(self, slines, 
                    min_length,
                    length_penalty, 
                    max_new_tokens, 
                    early_stopping,
                    temperature,
                    top_k, top_p,
                    no_repeat_ngram_size,
                    num_beams,
                    do_sample):
        
        with torch.no_grad():
            tokenized_slines = self.tokenizer(slines, 
                                              return_tensors='pt', 
                                              padding=True, truncation=True,
                                              max_length=self.stage_cfg.input_max_token)
            input = tokenized_slines.input_ids.to(self.device)
            
            hypotheses_batch = self.model.generate(input,
                                                   do_sample=do_sample,
                                                   decoder_start_token_id=self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 5,
                                                   eos_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 6,
                                                   min_length=min_length,
                                                   length_penalty=length_penalty,
                                                   early_stopping=early_stopping,
                                                   temperature=temperature,
                                                   top_k=top_k, top_p=top_p,
                                                   num_beams=num_beams,
                                                   max_new_tokens=max_new_tokens,
                                                   no_repeat_ngram_size=no_repeat_ngram_size
                                                   )
            free_memory(input)
            hypotheses_batch_decoded = [self.tokenizer.decode(hypotheses_batch[i].to("cpu"), skip_special_tokens=True)
                                        for i in range(hypotheses_batch.shape[0])]
            return hypotheses_batch_decoded

    def inference(self, 
                  bsz=8,
                  min_length=10,
                  length_penalty=1.0, 
                  max_new_tokens=120, 
                  early_stopping=True,
                  temperature=0.7,
                  top_k=50, top_p=0.8,
                  no_repeat_ngram_size=2,
                  num_beams=4,
                  do_sample=True) -> Dict[str, list]:
        
        logger.info(f"Inference settings: \n min_length{min_length} max_new_tokens{max_new_tokens} temperature{temperature} \ntop_k {top_k} top_p {top_p} no_repeat_ngram_size {no_repeat_ngram_size} num_beams {num_beams}")
        
        for data_type in ['train', 'val', 'test']:

            # we only need the test results on the last stage
            if data_type != 'test' and (self.fine_grained or self.test_mode):
                continue

            count = 1
            slines = [self.data[data_type][0]]
            for sline in tqdm(self.data[data_type][1:], desc=f"Generating {data_type} summaries"):
                if count % bsz == 0:
                    hypotheses_batch = self.run_summary(slines,
                                                        min_length,
                                                        length_penalty, 
                                                        max_new_tokens, 
                                                        early_stopping,
                                                        temperature,
                                                        top_k, top_p,
                                                        no_repeat_ngram_size,
                                                        num_beams,
                                                        do_sample)

                    for hypothesis in hypotheses_batch:
                        self.hypo[data_type].append(hypothesis)
                    slines = []

                slines.append(sline.strip())
                count += 1

            if slines:
                hypotheses_batch = self.run_summary(slines,
                                                    min_length,
                                                    length_penalty, 
                                                    max_new_tokens, 
                                                    early_stopping,
                                                    temperature,
                                                    top_k, top_p,
                                                    no_repeat_ngram_size,
                                                    num_beams,
                                                    do_sample)

                for hypothesis in hypotheses_batch:
                    self.hypo[data_type].append(hypothesis)

        return self.hypo

    def save(self):
        for data_type in ['train', 'val', 'test']:

            # we only need the test results on the last stage
            if data_type != 'test' and (self.fine_grained or self.test_mode):
                continue

            data_folder = os.path.join(self.cfg.train.output_path, f"stage_{self.cfg.cur_stage}")
            save_path = os.path.join(data_folder, f"{data_type}_split.hypo" if not self.fine_grained
            else f"{data_type}.hypo")
            write_list_asline(save_path, self.hypo[data_type])
