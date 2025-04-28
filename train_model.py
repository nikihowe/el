import os
import torch
import sys
import math # Needed for math.exp for perplexity if calculated manually
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator, # Can use this for eval sometimes
)
import wandb

# --- Wandb Logging Setup ---
wandb.init(project="el_takehome")

# --- Hardcoded Configuration --- (MODIFY THESE VALUES)
config_values = {
    # Model & Tokenizer
    "model_name": "EleutherAI/pythia-70m",
    "block_size": 1024,

    # Data
    "dataset_path": "arxiv-metadata-oai-snapshot.jsonl",
    "text_field": "abstract",
    "validation_split_percentage": 5, # Percentage of data to hold out for validation
    "preprocessing_num_workers": 8, # Reduced from os.cpu_count()
    "overwrite_cache": False,
    "dataloader_num_workers": 4,

    # Training
    "output_dir": "./pythia-70m-arxiv-scratch",
    "overwrite_output_dir": False,
    "num_train_epochs": 3.0,  # Increased epochs for scratch training
    "per_device_train_batch_size": 8,  # Increased batch size
    "gradient_accumulation_steps": 4,  # Adjusted for larger batch size
    "gradient_checkpointing": True,
    "learning_rate": 5e-6,  # Much lower learning rate for scratch training
    "weight_decay": 0.01,
    "warmup_steps": 5000,  # Much longer warmup for scratch training
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",

    # Evaluation & Logging
    "eval_strategy": "steps", # Evaluate periodically during training
    "eval_steps": 500, # Evaluate every N steps (match save_steps?)
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 50,
    "report_to": "wandb",
    "fp16": torch.cuda.is_available(),
}
# --- End Hardcoded Configuration ---

# Basic check for dataset file existence
if not os.path.exists(config_values["dataset_path"]):
    print(f"Error: Dataset file not found at {config_values['dataset_path']}")
    sys.exit(1)

# Check CUDA availability for fp16
if config_values["fp16"] and not torch.cuda.is_available():
     print("Warning: fp16 is enabled in config, but CUDA is not available. Training will proceed without fp16.")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config_values["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer pad_token to eos_token")

print(f"Loading configuration for {config_values['model_name']}...")
model_config = AutoConfig.from_pretrained(
    config_values["model_name"],
    trust_remote_code=True
)

print(f"Initializing {config_values['model_name']} model from scratch...")
model = AutoModelForCausalLM.from_config(model_config)
model.resize_token_embeddings(len(tokenizer))

print(f"Loading dataset: {config_values['dataset_path']}...")
# Load the JSONL dataset (expecting a 'train' split by default)
raw_dataset = load_dataset('json', data_files=config_values["dataset_path"])

# Split the dataset into training and validation
if "train" in raw_dataset:
    split_dataset = raw_dataset["train"].train_test_split(
        test_size=config_values['validation_split_percentage'] / 100.0,
        seed=42 # for reproducibility
    )
    # Rename 'test' split to 'validation' for clarity
    split_dataset['validation'] = split_dataset.pop('test')
    print(f"Split dataset into {100-config_values['validation_split_percentage']}% train and {config_values['validation_split_percentage']}% validation.")
else:
    print("Warning: 'train' split not found in dataset. Using the entire dataset for training.")
    # If no 'train' split, create a dummy DatasetDict for consistency
    # You might need to adjust this logic based on your actual dataset structure
    split_dataset = DatasetDict({"train": raw_dataset})
    config_values["eval_strategy"] = "no" # Disable evaluation if no split
    print("Disabling evaluation as no validation split could be created.")


# --- Tokenization Function ---
def tokenize_function(examples):
    texts = examples.get(config_values["text_field"])
    if texts is None:
        raise ValueError(f"Field '{config_values['text_field']}' not found in dataset.")
    valid_texts = [text for text in texts if isinstance(text, str)]
    if len(valid_texts) < len(texts):
        print(f"Warning: Skipped {len(texts) - len(valid_texts)} non-string or None entries in field '{config_values['text_field']}'.")
    return tokenizer(valid_texts)
# --- End Tokenization Function ---

print("Applying tokenization...")
tokenized_datasets = split_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=config_values["preprocessing_num_workers"],
    remove_columns=next(iter(split_dataset.values())).column_names, # Get columns from first split
    load_from_cache_file=not config_values["overwrite_cache"],
    desc="Running tokenizer on dataset splits",
)

# --- Block Processing Function ---
block_size = config_values["block_size"]
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
# --- End Block Processing Function ---

print(f"Grouping texts into blocks of size {block_size}...")
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=config_values["preprocessing_num_workers"],
    load_from_cache_file=not config_values["overwrite_cache"],
    desc=f"Grouping texts into chunks of {block_size}",
)

# Assign train and validation datasets
train_dataset = lm_datasets["train"]
if "validation" in lm_datasets:
    eval_dataset = lm_datasets["validation"]
    print(f"Number of training examples (blocks): {len(train_dataset)}")
    print(f"Number of validation examples (blocks): {len(eval_dataset)}")
else:
    eval_dataset = None
    print(f"Number of training examples (blocks): {len(train_dataset)}")
    print("No validation dataset available.")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=config_values["output_dir"],
    overwrite_output_dir=config_values["overwrite_output_dir"],
    num_train_epochs=config_values["num_train_epochs"],
    per_device_train_batch_size=config_values["per_device_train_batch_size"],
    gradient_accumulation_steps=config_values["gradient_accumulation_steps"],
    gradient_checkpointing=config_values["gradient_checkpointing"],
    # Evaluation args
    eval_strategy=config_values["eval_strategy"],
    eval_steps=config_values["eval_steps"] if config_values["eval_strategy"] != "no" else None,
    # Save args
    save_steps=config_values["save_steps"],
    save_total_limit=config_values["save_total_limit"],
    load_best_model_at_end=True if eval_dataset is not None else False, # Load best model based on eval
    metric_for_best_model="loss" if eval_dataset is not None else None,
    greater_is_better=False if eval_dataset is not None else None,
    # Logging args
    logging_steps=config_values["logging_steps"],
    report_to=config_values["report_to"],
    # Other args
    learning_rate=config_values["learning_rate"],
    weight_decay=config_values["weight_decay"],
    warmup_steps=config_values["warmup_steps"],
    lr_scheduler_type=config_values["lr_scheduler_type"],
    dataloader_num_workers=config_values["dataloader_num_workers"],
    max_grad_norm=config_values["max_grad_norm"],
)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # Pass validation dataset here
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
train_result = trainer.train()

print("Training finished. Saving final model (best checkpoint if evaluated)...")
trainer.save_model()  # Saves the best model if load_best_model_at_end=True
trainer.save_state()

# Log final metrics
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Evaluate at the end and log/save metrics
if eval_dataset is not None:
    print("*** Evaluate Final Model ***")
    eval_metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    eval_metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

print(f"\nTraining complete. Model saved to {config_values['output_dir']}")
