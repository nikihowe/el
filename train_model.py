import os
import torch
import sys
import math # Needed for math.exp for perplexity if calculated manually
from datasets import load_dataset, DatasetDict, load_from_disk
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

# Set cache directory to scratch
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.getcwd(), "hf_cache")

# --- Wandb Logging Setup ---
wandb.init(project="el_takehome")

# --- Hardcoded Configuration --- (MODIFY THESE VALUES)
config_values = {
    # Model & Tokenizer
    "model_name": "EleutherAI/pythia-70m",
    "block_size": 2048,  # Match Pythia's context window size

    # Data
    "dataset_path": "arxiv-metadata-oai-snapshot.jsonl",
    "text_field": "abstract",
    "validation_split_percentage": 5, # Percentage of data to hold out for validation
    "preprocessing_num_workers": 8, # Back to original value
    "overwrite_cache": False,
    "dataloader_num_workers": 4, # Back to original value
    "map_batch_size": 1000, # Keep this for efficiency

    # Training
    "output_dir": "./pythia-70m-arxiv-scratch",
    "overwrite_output_dir": False,
    "num_train_epochs": 3.0,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "learning_rate": 1e-7,  # Even more conservative
    "weight_decay": 0.01,
    "warmup_steps": 20000,  # Much longer warmup
    "max_grad_norm": 0.25,  # More aggressive clipping
    "lr_scheduler_type": "linear",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "fp16": True,
    "fp16_opt_level": "O2",  # More stable mixed precision
    "fp16_full_eval": False,

    # Evaluation & Logging
    "eval_strategy": "steps", # Evaluate periodically during training
    "eval_steps": 500, # Evaluate every N steps (match save_steps?)
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 50,
    "report_to": "wandb",
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
    trust_remote_code=True,
    use_cache=False,  # Explicitly disable caching for gradient checkpointing
)

# Ensure the model is configured for causal language modeling
model_config.is_decoder = True
model_config.add_cross_attention = False

print(f"Initializing {config_values['model_name']} model from scratch...")
# from_config without pretrained weights initializes the model with random weights
model = AutoModelForCausalLM.from_config(model_config)
model.config.use_cache = False  # Explicitly disable caching

# Apply GPT-J initialization
print("\nApplying GPT-J initialization...")
def gptj_init(module):
    if isinstance(module, (torch.nn.Linear,)):
        # GPT-J uses a scaled initialization
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)

# Apply initialization to all modules
model.apply(gptj_init)

# Test a single forward pass
print("\nTesting single forward pass...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model device: {next(model.parameters()).device}")

test_input = tokenizer("This is a test.", return_tensors="pt")
print(f"Input device before moving: {test_input['input_ids'].device}")
test_input = {k: v.to(device) for k, v in test_input.items()}
print(f"Input device after moving: {test_input['input_ids'].device}")

# Add labels for loss calculation
test_input["labels"] = test_input["input_ids"].clone()
print(f"Test input keys: {test_input.keys()}")

with torch.no_grad():
    outputs = model(**test_input)
    print(f"Outputs type: {type(outputs)}")
    print(f"Outputs attributes: {dir(outputs)}")
    if hasattr(outputs, 'loss'):
        print(f"Test forward pass loss: {outputs.loss.item():.6f}")
    else:
        print("No loss attribute found in outputs")
        # Try to compute loss manually if needed
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = test_input["labels"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print(f"Manually computed loss: {loss.item():.6f}")

model.resize_token_embeddings(len(tokenizer))

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

print(f"Loading dataset: {config_values['dataset_path']}...")
# Load the JSONL dataset (expecting a 'train' split by default)
raw_dataset = load_dataset('json', data_files=config_values["dataset_path"])

# Inspect raw data
print("\nInspecting raw data examples:")
for i in range(3):
    example = raw_dataset["train"][i]
    print(f"\nExample {i}:")
    print(f"Abstract: {example[config_values['text_field']][:200]}...")  # First 200 chars
    print(f"Length: {len(example[config_values['text_field']])}")

# Try to load saved processed datasets
try:
    print("\nAttempting to load saved processed datasets...")
    # Debug: Print current directory and check paths
    print(f"Current working directory: {os.getcwd()}")
    processed_path = "processed_datasets"
    print(f"Processed datasets path: {processed_path}")
    print(f"Path exists: {os.path.exists(processed_path)}")
    if os.path.exists(processed_path):
        print(f"Directory contents: {os.listdir(processed_path)}")
    
    if os.path.exists(processed_path):
        # Load using load_from_disk
        lm_datasets = load_from_disk(processed_path)
        print("Successfully loaded saved processed datasets!")
        
        # Inspect processed data
        print("\nInspecting processed data examples:")
        for i in range(3):
            example = lm_datasets["train"][i]
            print(f"\nExample {i}:")
            print(f"Input IDs length: {len(example['input_ids'])}")
            print(f"First 50 tokens: {tokenizer.decode(example['input_ids'][:50])}")
            print(f"Labels length: {len(example['labels'])}")
            print(f"First 50 labels: {tokenizer.decode(example['labels'][:50])}")
    else:
        raise FileNotFoundError("Saved datasets not found")
except Exception as e:
    print(f"Could not load saved datasets: {e}")
    print("Processing datasets from scratch...")
    
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
        split_dataset = DatasetDict({"train": raw_dataset})
        config_values["eval_strategy"] = "no"
        print("Disabling evaluation as no validation split could be created.")

    print("Applying tokenization...")
    # Process each split separately with caching
    tokenized_datasets = {}
    for split_name, dataset in split_dataset.items():
        print(f"Tokenizing {split_name} split...")
        tokenized_datasets[split_name] = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=config_values["map_batch_size"],
            num_proc=config_values["preprocessing_num_workers"],
            remove_columns=dataset.column_names,
            load_from_cache_file=not config_values["overwrite_cache"],
            desc=f"Running tokenizer on {split_name} split",
            writer_batch_size=config_values["map_batch_size"],
        )
    tokenized_datasets = DatasetDict(tokenized_datasets)

    # Save tokenized datasets
    tokenized_datasets.save_to_disk("tokenized_datasets")

    print(f"Grouping texts into blocks of size {block_size}...")
    # Process each split separately with caching
    lm_datasets = {}
    for split_name, dataset in tokenized_datasets.items():
        print(f"Grouping {split_name} split...")
        lm_datasets[split_name] = dataset.map(
            group_texts,
            batched=True,
            batch_size=config_values["map_batch_size"],
            num_proc=config_values["preprocessing_num_workers"],
            load_from_cache_file=not config_values["overwrite_cache"],
            desc=f"Grouping {split_name} texts into chunks of {block_size}",
            writer_batch_size=config_values["map_batch_size"],
        )
    lm_datasets = DatasetDict(lm_datasets)

    # Save processed datasets
    lm_datasets.save_to_disk("processed_datasets")

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
