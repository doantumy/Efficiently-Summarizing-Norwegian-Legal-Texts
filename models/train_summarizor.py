import os

import wandb
from loguru import logger
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    DataCollatorForSeq2Seq, GenerationConfig
from datasets import DatasetDict, Dataset
import evaluate
import torch
from utils.parameters import parse_parameters
from utils.gpu import free_memory


def load_file_pair(base_path: str, file_name: str):
    source_file = f'{base_path}/{file_name}.source'
    logger.info(f"Loading {source_file} & target")

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Data {source_file} not found")

    with open(source_file, 'r') as file:
        source = file.readlines()
    with open(f'{base_path}/{file_name}.target', 'r') as file:
        target = file.readlines()

    # change format to hugging_face format
    data = [{'summary': t, 'text': s} for s, t in zip(source, target)]
    dataset = Dataset.from_list(data)  # , features={'text': 'string', 'label': 'string'}
    return dataset


def load_data(base_path: str):
    logger.info("Loading source and data, this may take a while...")

    train = load_file_pair(base_path, 'train')
    val = load_file_pair(base_path, 'val')
    test = load_file_pair(base_path, 'test')

    dataset = {'train': train, 'val': val, 'test': test}
    dataset = DatasetDict(dataset)

    return dataset


class Tokenizer:
    def __init__(self, tokenizer=None, max_length=None):
        if tokenizer is None:
            logger.info("Using default tokenizer: ltg/nort5-small")
            self.tokenizer = AutoTokenizer.from_pretrained("ltg/nort5-small")
        elif isinstance(tokenizer, str):
            logger.info(f"Using tokenizer: {tokenizer}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            logger.info("Using custom tokenizer")
            self.tokenizer = tokenizer

        self.max_length = max_length

    def _preprocess(self, exapmle):
        prefix = "oppsummer: "
        inputs = [prefix + doc for doc in exapmle["text"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_length, truncation=True)
        labels = self.tokenizer(exapmle["summary"], max_length=self.max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __call__(self, dataset: DatasetDict):
        return dataset.map(self._preprocess, batched=True)

    def get_tokenizer(self):
        return self.tokenizer


class T5FineTuner:
    def __init__(self, model=None, 
                 tokenizer=None, 
                 output_dir=None, 
                 warmup_steps=None, 
                 epochs=None,
                 learning_rate=None,
                 batch_size=None,
                 save_total_limit=None,
                 logging_steps=None, 
                 ):
        if torch.cuda.is_available():
            self.device = "cuda"
            use_cpu = False
        else:
            self.device = "cpu"
            use_cpu = True
        logger.info(f"Using device: {self.device}")

        if model is None:
            logger.info("Using default model: ltg/nort5-small")
            default_model = "ltg/nort5-small"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(default_model,
                                                               trust_remote_code=True).to(self.device)
        elif isinstance(model, str):
            logger.info(f"Using model: {model}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model,
                                                               trust_remote_code=True).to(self.device)
        else:
            logger.info("Using custom model")
            self.model = model

        self.tokenizer = tokenizer.get_tokenizer()

        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model)
        self.rouge = evaluate.load("rouge")

        self.output = output_dir
        self.push_to_hub = False

        self.generation_config = GenerationConfig(
            max_length=self.model.config.max_length,
            max_new_tokens=self.model.config.max_new_tokens,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
        )

        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.output,
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            predict_with_generate=True,
            push_to_hub=self.push_to_hub,
            use_cpu=use_cpu,
            generation_config=self.generation_config,
            report_to="wandb",
            logging_strategy="steps", 
            logging_steps=logging_steps,
            # fp16=torch.cuda.is_available(),
        )


    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def __call__(self, dataset):
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        logger.info("Training model...")
        self.trainer.train()
        logger.info(f"Training finished, saving model to {self.output}")
        self.trainer.save_model(self.output)


# implement main function to run all
def main(args):
    logger.info(f"Current dir: {os.getcwd()}")
    input_path = args.input_data_path
    output_path = args.output_model_path
    dataset = load_data(input_path)
    logger.info(f"Train summarizor args: {args}")
    base_model = args.model

    config = {
        "input_data_path": args.input_data_path,
        "output_model_path": args.output_model_path,
        "model": args.model,
        "warmup": args.warmup,
        "epochs": args.epochs,
        "run": args.run,
        "group_name": args.group_name,
        "wandb_mode": args.wandb_mode,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "max_length": args.max_length,
    }
    stage = input_path.split("/")[-1]

    wandb.init(project=args.run, group=args.group_name, config=config, mode=args.wandb_mode,
               save_code=False, name=stage)

    tokenizer = Tokenizer(tokenizer=base_model, max_length=args.max_length)
    tokenized_data = tokenizer(dataset)

    t5_fine_tuner = T5FineTuner(model=base_model, 
                                tokenizer=tokenizer, 
                                output_dir=output_path,
                                warmup_steps=args.warmup, 
                                epochs=args.epochs,
                                learning_rate=args.learning_rate,
                                batch_size=args.batch_size,
                                save_total_limit=args.save_total_limit,
                                logging_steps=args.logging_steps,)
    t5_fine_tuner(tokenized_data)

    logger.info("Training finished")
    wandb.finish()
    free_memory([t5_fine_tuner], debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_data_path", type=str, default="../output/AMI/stage_0",
                        help="Path to the input data")
    parser.add_argument("-o", "--output_model_path", type=str, default="../output/AMI/stage_1/trainer_output",
                        help="Path to the output data")
    parser.add_argument("-m", "--model", type=str, default="ltg/nort5-small",
                        help="Path to the output data")
    parser.add_argument("-w", "--warmup", type=int, default=0)
    parser.add_argument("-e", "--epochs", type=int, default=4)
    parser.add_argument("-r", "--run", type=str, default="summ.n")
    parser.add_argument("-g", "--group-name", type=str, default="default")
    parser.add_argument("-wm", "--wandb-mode", type=str, default="offline")
    
    logger.info(f"Loading geneneration settings parameters")
    # Add the parameters from utils/parameters.py
    parser = parse_parameters(parser)
    logger.info("GPU status in model finetuner:")
    os.system("nvidia-smi")
    main(parser.parse_args())