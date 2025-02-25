import argparse
import os
import datetime
import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from loguru import logger
import wandb
import random

def load_data(input_path, debug=False):
    """
    Loads and preprocesses the dataset.

    Args:
        input_path (str): Path to the dataset JSON file.
        debug (bool): If True, reduce data size by 10x for debugging.

    Returns:
        list: Processed dataset.
    """
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    if debug:
        data = data[: len(data) // 10]
        logger.info("Debug mode enabled: Reduced dataset size by 10x.")
    
    return data

def preprocess_data(data):
    """
    Prepares the data for regression.

    Args:
        data (list): List of dictionaries with 'summary', 'text', and 'label'.

    Returns:
        list: Processed dataset.
    """
    return [{"input_text": item["text"], "label": item["label"]} for item in data]

def main(args):
    logger.info(f"Current dir: {os.getcwd()}")

    # Add date-time into saved output folder
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_model_path}regression.cls.{date_time}/trainer_output"
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output path: {output_path}")

    # Load datasets
    logger.info("Loading train, validation, and test datasets.")
    train_data = load_data(args.train_data_path, args.debug)
    val_data = load_data(args.val_data_path, args.debug)
    test_data = load_data(args.test_data_path, args.debug)

    # Preprocess datasets
    logger.info("Preprocessing datasets.")
    train_data = preprocess_data(train_data)
    val_data = preprocess_data(val_data)
    test_data = preprocess_data(test_data)

    # Shuffle the datasets with a seed
    logger.info(f"Shuffling datasets with seed {args.random_seed}.")
    random.seed(args.random_seed)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Convert to Hugging Face dataset
    train_dataset = Dataset.from_dict({
        "text": [d["input_text"] for d in train_data],
        "label": [d["label"] for d in train_data]
    })
    val_dataset = Dataset.from_dict({
        "text": [d["input_text"] for d in val_data],
        "label": [d["label"] for d in val_data]
    })
    test_dataset = Dataset.from_dict({
        "text": [d["input_text"] for d in test_data],
        "label": [d["label"] for d in test_data]
    })

    datasets = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})

    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=args.max_length)
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])  # Keep only tokenized inputs and labels
    tokenized_datasets.set_format("torch")

    # Initialize wandb logging if enabled
    if args.do_logging:
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "warmup_steps": args.warmup,
                "max_length": args.max_length,
            },
        )
        logger.info(f"Wandb logging enabled - Project: {args.project_name}, Run: {args.run_name}")
    report_to_platforms = ["wandb"] if args.do_logging else ["none"]

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        logging_dir=f"{output_path}/logs",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=3,  # Keep only the 3 most recent checkpoints
        report_to=report_to_platforms,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use evaluation loss to determine the best model
        greater_is_better=False  # Smaller loss is better for regression tasks
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],  # Add validation dataset
        tokenizer=tokenizer,
    )

    # Train model
    if args.do_training:
        trainer.train()

    # Evaluate model
    if args.do_testing:
        # Evaluate on test dataset
        test_loss_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        logger.info(f"Test loss results: {test_loss_results}")

        # Generate predictions for the test set
        predictions = trainer.predict(tokenized_datasets["test"])
        predicted_labels = predictions.predictions.squeeze().tolist()

        # Prepare results for saving
        test_results = []
        for _, (text, label, pred) in enumerate(zip(test_data, tokenized_datasets["test"]["label"], predicted_labels)):
            test_results.append({
                "text": text["input_text"],
                "true_label": float(label),
                "predicted_label": float(pred),
            })

        # Save test results to JSON
        test_results_path = os.path.join(output_path, "test_results.json")
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)
        logger.info(f"Test results saved to {test_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train-data-path", 
                        type=str, 
                        required=True, 
                        help="Path to the train data file")
    parser.add_argument("-val", "--val-data-path", 
                        type=str, 
                        required=True, 
                        help="Path to the validation data file")
    parser.add_argument("-test", "--test-data-path", 
                        type=str, 
                        required=True, 
                        help="Path to the test data file")
    parser.add_argument("-o", "--output-model-path", 
                        type=str, 
                        required=True, 
                        default="./output/",
                        help="Path to save model outputs")
    parser.add_argument("-m", "--model", type=str, default="NbAiLab/nb-bert-base", help="Pretrained model")
    parser.add_argument("-lr", "--learning-rate", type=float, default=2e-5)
    parser.add_argument("-w", "--warmup", type=int, default=100)
    parser.add_argument("-bs", "--batch-size", type=int, default=8)
    parser.add_argument("-ml", "--max-length", type=int, default=512)
    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-wd", "--weight-decay", type=float, default=0.01)
    parser.add_argument("-l", "--do-logging", action="store_true", default=False)
    parser.add_argument("-p", "--project-name", type=str, default="regression_classifier")
    parser.add_argument("-r", "--run-name", type=str, default="no_regression_classifier")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Reduce dataset size for debugging")
    parser.add_argument("-dt", "--do-testing", action="store_true", default=False)
    parser.add_argument("-dr", "--do-training", action="store_true", default=False)
    parser.add_argument("-ls", "--logging-steps", type=int, default=10, help="Frequency of logging steps during training")
    parser.add_argument("-rs", "--random-seed", type=int, default=123, help="Random seed for shuffling data")
    args = parser.parse_args()

    main(args)