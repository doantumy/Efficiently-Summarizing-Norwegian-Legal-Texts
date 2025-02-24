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
import json
import datetime

def load_file_pair(base_path: str, file_name: str, debug: bool = False):
    source_file = f'{base_path}/{file_name}.source'
    logger.info(f"Loading {source_file} & target")

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Data {source_file} not found")

    with open(source_file, 'r') as file:
        source = file.readlines()
    with open(f'{base_path}/{file_name}.target', 'r') as file:
        target = file.readlines()

    # Apply debug mode
    if debug:
        logger.info("Debug mode enabled: Reducing dataset size by 100x")
        source = source[:len(source)//100]
        target = target[:len(target)//100]
        logger.info(f"Number of sentences in debug mode: {len(source)}")

    # Change format to Huggingface format
    data = [{'summary': t, 'text': s} for s, t in zip(source, target)]
    dataset = Dataset.from_list(data)
    return dataset


def load_data(base_path: str, debug: bool = False):
    logger.info("Loading source and data, this may take a while...")

    train = load_file_pair(base_path, 'train', debug=debug)
    val = load_file_pair(base_path, 'val', debug=debug)
    test = load_file_pair(base_path, 'test', debug=debug)

    dataset = {'train': train, 'val': val, 'test': test}
    dataset = DatasetDict(dataset)

    return dataset


class Tokenizer:
    def __init__(self, tokenizer=None, max_length=512):
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
        logger.info(f"max_length: {max_length}")


    def _preprocess(self, example):
        prefix = "oppsummer: "
        inputs = [prefix + doc for doc in example["text"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_length, truncation=True)
        labels = self.tokenizer(example["summary"], max_length=self.max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __call__(self, dataset: DatasetDict):
        return dataset.map(self._preprocess, batched=True)

    def get_tokenizer(self):
        return self.tokenizer


class T5FineTuner:
    def __init__(self, model=None, tokenizer=None, output_dir=None, train_config_dict=None):
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
            self.model_path = default_model
            
        elif isinstance(model, str):
            logger.info(f"Using model: {model}")
            self.model_path = model
           
        else:
            logger.info("Using custom model")
            self.model = model
            self.model_path = None

        if self.model_path:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)

        self.tokenizer = tokenizer.get_tokenizer()

        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model)
        self.rouge = evaluate.load("rouge")

        self.output = output_dir
        self.push_to_hub = False

        self.generation_config = GenerationConfig(
            max_length=train_config_dict.get("max_length", 512),  # Use the max_length from the config or default to 512
            max_new_tokens=train_config_dict.get("max_new_tokens", 256),
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
        )

        self.train_config_dict = train_config_dict
        self.do_training = self.train_config_dict["do_training"]
        self.do_validation = self.train_config_dict["do_validation"]
        self.do_testing = self.train_config_dict["do_testing"]
        self.do_logging = self.train_config_dict["do_logging"]

        # Conditionally set the report_to value
        self.report_to = "wandb" if self.train_config_dict["do_logging"] else "none"
        logger.info(f"Logging train/val/test to: {self.report_to}")
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.output,
            eval_strategy="epoch",
            learning_rate=self.train_config_dict["learning_rate"],
            per_device_train_batch_size=self.train_config_dict["batch_size"],
            per_device_eval_batch_size=self.train_config_dict["batch_size"],
            weight_decay=self.train_config_dict["weight_decay"],
            save_strategy="epoch",
            save_total_limit=self.train_config_dict["save_total_limit"],
            load_best_model_at_end=True,
            warmup_steps=self.train_config_dict["warmup_steps"],
            num_train_epochs=self.train_config_dict["epochs"],
            predict_with_generate=True,
            # fp16=True,
            push_to_hub=self.push_to_hub,
            use_cpu=use_cpu,
            generation_config=self.generation_config,
            report_to=self.report_to,
            logging_strategy="steps", logging_steps=10,
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
    

    def save_test_results(self, test_results, output_dir):
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)
        logger.info(f"Saved test results to {os.path.join(output_dir, 'test_results.json')}")


    def save_summaries(self, summaries, output_dir):
        with open(os.path.join(output_dir, "generated_summaries.txt"), "w") as f:
            for summary in summaries:
                f.write(summary + "\n")
        logger.info(f"Saved generated summaries to {os.path.join(output_dir, 'generated_summaries.txt')}")


    def generate_summaries(self, dataset, generation_params=None):
        generation_params = generation_params or {}
        summaries = []
        labels = []
        
        generation_params = {**self.generation_config.to_dict(), **generation_params}
        logger.info(f"Generation parameters: {generation_params}")

        for example in dataset:
            # Prepare input_ids and labels
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(self.device)
            labels.append(example["summary"])

            # Generate summaries
            sum_ids = self.model.generate(input_ids, **generation_params)
            summary = self.tokenizer.decode(sum_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        return labels, summaries
    

    def __call__(self, dataset, generation_params=None):
        train_dataset = dataset["train"] if self.do_training else None
        eval_dataset = dataset["val"] if self.do_validation else None
        test_dataset = dataset["test"] if self.do_testing else None
        if self.do_training :
            self.trainer = Seq2SeqTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )
            logger.info("Training model...")
            self.trainer.train()
            logger.info(f"Training finished, saving model to {self.output}")
            self.trainer.save_model(self.output)

            # Load the best model after training
            logger.info("Loading the best model after training...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.output, trust_remote_code=True).to(self.device)
        
        elif not self.do_training and self.model_path:
            # Load the provided model path if not training
            logger.info(f"Loading model from path: {self.model}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)

        if self.do_testing and test_dataset is not None:
            logger.info("Running test evaluation and summary generation...")
            # Generate summaries for the test set
            labels, summaries = self.generate_summaries(test_dataset, generation_params)

            # Compute metrics on the generated summaries
            logger.info("Computing metrics...")
            result = self.rouge.compute(predictions=summaries, references=labels, use_stemmer=True)

            test_results = {k: round(v, 4) for k, v in result.items()}
            logger.info(f"Test results: {test_results}")
            # Save results and summaries
            self.save_test_results(test_results, self.output)
            self.save_summaries(summaries, self.output)


# Implement main function to run all
def main(args):
    logger.info(f"Current dir: {os.getcwd()}")
    
    input_path = args.input_data_path
    logger.info(f"Input path: {input_path}")

    # Add date time into saved output folder
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_model_path}/{args.dataset_name}.classic.{date_time}/trainer_output"
    logger.info(f"Output path: {output_path}")
    dataset = load_data(input_path, args.debug)

    train_config_dict = {
                         "max_length": args.max_length,
                         "learning_rate":args.learning_rate,
                         "warmup_steps": args.warmup,
                         "save_total_limit": args.save_total_limit,
                         "batch_size": args.batch_size,
                         "epochs": args.epochs,
                         "weight_decay": args.weight_decay,
                         "do_logging": args.do_logging,
                         "project_name": args.project_name,
                         "run_name": f"{args.run_name}_{date_time}",
                         "do_training": args.do_training,
                         "do_validation": args.do_validation,
                         "do_testing": args.do_testing}
    logger.info(f"Training configuration: {train_config_dict}")
    
    generation_params_dict = {
            "num_beams": args.num_beams, 
            "early_stopping": True,
            "do_sample": True,
            "min_length": args.min_length,
            "temperature": args.temperature,
            "top_k": args.top_k, 
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "no_repeat_ngram_size": args.no_repeat_ngram_size
        }
    
    logger.info(f"Generation parameters: {generation_params_dict}")

    # Initialize wandb logging if enabled
    if train_config_dict["do_logging"]:
        project_name = train_config_dict["project_name"]
        run_name = train_config_dict["run_name"]
        logger.info(f"Logging training with wandb - Project name: {project_name}, Run name: {run_name}")
        wandb.init(
        project=project_name,
        name=run_name,
        config={
            "learning_rate": train_config_dict["learning_rate"],
            "epochs": train_config_dict["epochs"],
            "batch_size": train_config_dict["batch_size"],
            "warmup_steps": train_config_dict["warmup_steps"] 
            }
        )

    base_model = args.model

    tokenizer = Tokenizer(tokenizer=base_model, max_length=train_config_dict["max_length"])
    tokenized_data = tokenizer(dataset)

    t5_fine_tuner = T5FineTuner(model=base_model, tokenizer=tokenizer, 
                                output_dir=output_path, train_config_dict=train_config_dict)
    
    # Pass the tokenized data to T5FineTuner
    t5_fine_tuner(dataset=tokenized_data, generation_params=generation_params_dict)
    logger.info("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-data-path", type=str, default="./data/lovdata",
                        help="Path to the input data")
    parser.add_argument("-o", "--output-model-path", type=str, default="./output",
                        help="Path to the output data")
    parser.add_argument("-m", "--model", type=str, default="ltg/nort5-small",
                        help="Path to the output data")
    parser.add_argument("-dn", "--dataset-name", type=str, default="LOV"),
    parser.add_argument("-lr", "--learning-rate", type=float, default=2e-5)
    parser.add_argument("-w", "--warmup", type=int, default=100)
    parser.add_argument("-stl", "--save-total-limit", type=int, default=1)
    parser.add_argument("-bs", "--batch-size", type=int, default=8)
    parser.add_argument("-ml", "--max-length", type=int, default=512)
    parser.add_argument("-e", "--epochs", type=int, default=4)
    parser.add_argument("-wd", "--weight-decay", type=float, default=0.01)
    parser.add_argument("-l", "--do-logging", action="store_true", default=False)
    parser.add_argument("-p", "--project-name", type=str, default="classic.summarization")
    parser.add_argument("-r", "--run-name", type=str, default="classic.finetune")
    parser.add_argument("-dr", "--do-training", action="store_true", default=False)
    parser.add_argument("-dv", "--do-validation", action="store_true", default=False)
    parser.add_argument("-dt", "--do-testing", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False, 
                        help="Enable debug mode to reduce data size by 10x")
    parser.add_argument("-mnl", "--min-length", type=int, default=128, 
                        help="Geneneration settings: Min length of generated output.")
    parser.add_argument("-mnt", "--max-new-tokens", type=int, default=256, 
                        help="Geneneration settings: Max number of new tokens")
    parser.add_argument("-tem", "--temperature", type=float, default=0.7, 
                        help="Geneneration settings: Temperature < 1 means more deterministic output")
    parser.add_argument("-tk", "--top-k", type=int, default=40, 
                        help="Geneneration settings: Top k value, >50 means more creative output")
    parser.add_argument("-tp", "--top-p", type=float, default=0.8, 
                        help="Geneneration settings: Top p value, <=0.8 for deterministic output")
    parser.add_argument("-nrng", "--no-repeat-ngram-size", type=int, default=2, 
                        help="Geneneration settings: No repeat ngram size")
    parser.add_argument("-nb", "--num-beams", type=int, default=2, 
                        help="Geneneration settings: =1: greedy search, >1: beam search multiple hypotheses at each step")

    
    logger.info("GPU status in model finetuner:")
    os.system("nvidia-smi")
    main(parser.parse_args())

# python3 models/train_classic_summarizer.py --max-length 512 --do-training --do-validation --do-testing --epoch 3 --debug --do-logging --model ltg/nort5-large