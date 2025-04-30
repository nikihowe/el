import math
import os
import sys

import torch
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import wandb
from datasets import DatasetDict, load_dataset, load_from_disk

from debug_utils import test_forward_pass

# Original dataset
DATA_PATH = 'datasets/pretraining/arxiv-metadata-oai-snapshot.jsonl'
# Tokenized but not grouped
TOKENIZED_DATA_PATH = 'datasets/pretraining/tokenized_datasets'
# Finished processed dataset, ready for training
PROCESSED_DATA_PATH = 'datasets/pretraining/grouped_datasets'
LOG_TO_WANDB = False
DEBUG = True

# Set cache directory to `scratch` partition
# NOTE: This is necessary because the default HF cache writes to $HOME,
# which has a storage cap of 100GB and will fill up.
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
# Avoids a warning about avoiding deadlocks with num_workers > 0
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- Wandb Logging Setup ---
wandb.init(project='el_takehome')

# --- Hardcoded Configuration ---
config_values = {
    # Model & Tokenizer
    'model_name': 'EleutherAI/pythia-70m',
    'block_size': 2048,
    # Data
    'dataset_path': DATA_PATH,
    'text_field': 'abstract',
    'validation_split_percentage': 5,  # Percentage of data to hold out for validation
    'preprocessing_num_workers': 8,  # Going too high caused memory issues
    'overwrite_cache': False,
    'dataloader_num_workers': 4,  # Going too high caused memory issues
    'map_batch_size': 1000,
    # Training
    'output_dir': './pythia-70m-arxiv-scratch',
    'overwrite_output_dir': False,
    'num_train_epochs': 3.0,
    'per_device_train_batch_size': 4,
    'gradient_accumulation_steps': 8,
    'gradient_checkpointing': True,
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.05,  # Warmup period for stability early on
    'max_grad_norm': 1.0,
    'lr_scheduler_type': 'cosine',
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'adam_epsilon': 1e-6,  # NOTE: raising this from 1e-8 was necessary for stability
    'bf16': True,  # NOTE: fp16 caused `Attempting to unscale FP16 gradients` error
    # Evaluation & Logging
    'eval_strategy': 'steps',
    'eval_steps': 500,
    'save_steps': 500,
    'save_total_limit': 2,
    'logging_steps': 50,
    'report_to': 'wandb' if LOG_TO_WANDB else 'none',
    'seed': 42,
}
# --- End Hardcoded Configuration ---

# Possibly use `bf16` mixed precision for training
# NOTE: tried `fp16` but got `Attempting to unscale FP16 gradients` error
precision_mode = 'bf16' if config_values['bf16'] else 'no'

print(f"Initializing Accelerator with mixed_precision='{precision_mode}'")
accelerator = Accelerator(mixed_precision=precision_mode)
print(f'Using device: {accelerator.device}')

print('Loading tokenizer...')
# Explicitly add pad token if missing
tokenizer = AutoTokenizer.from_pretrained(config_values['model_name'])
if tokenizer.pad_token is None:
    print(
        'Tokenizer does not have a pad token. Adding eos_token as pad_token.'
    )
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading configuration for {config_values['model_name']}...")
model_config = AutoConfig.from_pretrained(
    config_values['model_name'],
    use_cache=False,  # Required for gradient checkpointing
    pad_token_id=tokenizer.pad_token_id,  # Ensure config knows pad token id
)

print(f"Initializing {config_values['model_name']} model from scratch...")
# NOTE: from_config without pretrained weights initializes the model with random weights
model = AutoModelForCausalLM.from_config(model_config)

# Resize token embeddings in case tokenizer vocab size differs from config (e.g., added pad token)
model.resize_token_embeddings(len(tokenizer))
print(f'Resized model token embeddings to: {len(tokenizer)}')

# --- Test a single forward pass using the utility function ---
# Move model to the correct device *before* testing
model = model.to(accelerator.device) 
# Call the testing function
if DEBUG:
    test_forward_pass(model, tokenizer, accelerator)

# --- Tokenization Function ---
def tokenize_function(examples):
    texts = examples.get(config_values['text_field'])
    if texts is None:
        raise ValueError(
            f"Field '{config_values['text_field']}' not found in dataset."
        )
    # Filter out None or non-string entries BEFORE tokenization
    valid_texts = [
        text for text in texts if isinstance(text, str) and text.strip()
    ]
    if len(valid_texts) < len(texts):
        print(
            f"Warning: Skipped {len(texts) - len(valid_texts)} non-string, empty or None entries in field '{config_values['text_field']}'."
        )
    # Tokenize valid texts. Padding handled later by collator.
    # Truncation might be needed if texts are very long.
    # We let group_texts handle chunking.
    return tokenizer(
        valid_texts,
        truncation=True,
        # NOTE: We do times 2 to ensure efficient packing
        max_length=config_values['block_size'] * 2,
        padding=False,
    )   # Added truncation safeguard

block_size = config_values['block_size']

def group_texts(examples):
    """Concatenates texts from a batch and chunks them into blocks of fixed size."""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        print(f"Dropping {total_length - block_size} tokens out of {total_length} total tokens")

    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    return result

# Try to load saved processed datasets
try:
    print(
        f'\nAttempting to load saved grouped datasets from {PROCESSED_DATA_PATH}...'
    )
    if os.path.exists(PROCESSED_DATA_PATH):
        lm_datasets = load_from_disk(PROCESSED_DATA_PATH)
        print('Successfully loaded saved grouped datasets!')

        # Verify expected columns are present
        if 'input_ids' not in lm_datasets['train'].column_names:
            raise ValueError("Loaded dataset missing 'input_ids' column.")

        print('\nInspecting processed data examples:')
        num_processed_to_show = min(3, len(lm_datasets['train']))
        for i in range(num_processed_to_show):
            example = lm_datasets['train'][i]
            print(f'\nExample {i}:')
            print(f"Input IDs length: {len(example['input_ids'])}")
            print(
                f"First 50 tokens: {tokenizer.decode(example['input_ids'][:50])}"
            )
    else:
        raise FileNotFoundError(f"Directory '{PROCESSED_DATA_PATH}' not found")

except Exception as e:
    print(
        f'Could not load saved grouped datasets: {e}. Processing from scratch...'
    )

    # Try loading tokenized cache first
    try:
        if not config_values['overwrite_cache'] and os.path.exists(
            TOKENIZED_DATA_PATH
        ):
            print(
                f'Attempting to load tokenized datasets from {TOKENIZED_DATA_PATH}...'
            )
            tokenized_datasets = load_from_disk(TOKENIZED_DATA_PATH)
            # Verify splits match
            if set(tokenized_datasets.keys()) != set(split_dataset.keys()):
                raise ValueError(
                    'Cached tokenized dataset splits do not match current splits.'
                )
            print('Successfully loaded cached tokenized datasets.')
        else:
            raise FileNotFoundError(
                'Tokenized cache not found or overwrite requested.'
            )
    except Exception as te:
        print(f'Could not load tokenized cache: {te}. Tokenizing from scratch...')

        # --- Load Raw Data Only If Caches Missed ---
        print(f"Loading raw dataset: {config_values['dataset_path']}...")
        # Basic check for dataset file existence
        if not os.path.exists(config_values['dataset_path']):
            print(f"Error: Dataset file not found at {config_values['dataset_path']}")
            sys.exit(1)
        try:
            raw_dataset = load_dataset(
                'json', data_files=config_values['dataset_path'], split='train'
            )
            raw_datasets = DatasetDict(
                {'train': raw_dataset}
            )
        except Exception as load_e:
            print(f'Error loading dataset: {load_e}')
            print(
                "Please ensure the jsonl file is correctly formatted and contains a 'train' structure or adjust split name."
            )
            sys.exit(1)

        # Inspect raw data (Optional, only if debugging)
        if DEBUG:
            print('\nInspecting raw data examples:')
            num_examples_to_show = min(3, len(raw_datasets['train']))
            for i in range(num_examples_to_show):
                example = raw_datasets['train'][i]
                abstract = example.get(
                    config_values['text_field'], 'N/A'
                )
                print(f'\nExample {i}:')
                print(f'Abstract: {abstract[:200]}...')
                print(f'Length: {len(abstract)}')

        # Split the raw dataset into training and validation
        if config_values['validation_split_percentage'] > 0:
            split_dataset = raw_datasets['train'].train_test_split(
                test_size=config_values['validation_split_percentage'] / 100.0,
                seed=config_values['seed'],
            )
            # Rename 'test' split to 'validation' for clarity
            split_dataset['validation'] = split_dataset.pop('test')
            print(
                f"Split dataset into {100-config_values['validation_split_percentage']}% train and {config_values['validation_split_percentage']}% validation."
            )
        else:
            print(
                'Validation split percentage is 0. Using the entire dataset for training.'
            )
            split_dataset = DatasetDict({'train': raw_datasets['train']})
            config_values[
                'eval_strategy'
            ] = 'no'
            print('Evaluation disabled as no validation split was created.')
        # --- End Raw Data Loading ---

        print('Applying tokenization...')
        tokenized_datasets = split_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=config_values['map_batch_size'],
            num_proc=config_values['preprocessing_num_workers'],
            remove_columns=next(
                iter(split_dataset.values())
            ).column_names,  # Remove original columns
            load_from_cache_file=not config_values['overwrite_cache'],
            desc='Running tokenizer on dataset splits',
            writer_batch_size=config_values['map_batch_size'],
        )
        print(f'Saving tokenized datasets to {TOKENIZED_DATA_PATH}...')
        tokenized_datasets.save_to_disk(TOKENIZED_DATA_PATH)

        print(f'Grouping texts into blocks of size {block_size}...')
        # Note: group_texts removes remaining columns ('attention_mask' potentially)
        # The collator will recreate attention_mask if needed based on padding
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=config_values[
                'map_batch_size'
            ],  # Process N tokenized entries to form blocks
            num_proc=config_values['preprocessing_num_workers'],
            load_from_cache_file=not config_values['overwrite_cache'],
            desc=f'Grouping texts into chunks of {block_size}',
            writer_batch_size=config_values[
                'map_batch_size'
            ],  # Control writing frequency
        )

        # Filter out empty examples potentially created by group_texts if total length < block_size
        for split in lm_datasets:
            initial_count = len(lm_datasets[split])
            lm_datasets[split] = lm_datasets[split].filter(
                lambda example: len(example['input_ids']) > 0
            )
            filtered_count = len(lm_datasets[split])
            if initial_count != filtered_count:
                print(
                    f"Filtered {initial_count - filtered_count} empty examples from '{split}' split."
                )

        print(f'Saving grouped datasets to {PROCESSED_DATA_PATH}...')
        lm_datasets.save_to_disk(PROCESSED_DATA_PATH)

# Assign train and validation datasets
if 'train' not in lm_datasets or len(lm_datasets['train']) == 0:
    print(
        'Error: No training data found after processing. Check dataset and processing steps.'
    )
    sys.exit(1)
train_dataset = lm_datasets['train']

if 'validation' in lm_datasets and len(lm_datasets['validation']) > 0:
    eval_dataset = lm_datasets['validation']
    print(f'Number of training examples (blocks): {len(train_dataset)}')
    print(f'Number of validation examples (blocks): {len(eval_dataset)}')
    # Ensure eval strategy is enabled if eval_dataset exists
    if config_values['eval_strategy'] == 'no':
        print(
            "Warning: Validation dataset exists but eval_strategy is 'no'. Evaluation will not run."
        )

else:
    eval_dataset = None
    print(f'Number of training examples (blocks): {len(train_dataset)}')
    print('No validation dataset available or it is empty.')
    config_values['eval_strategy'] = 'no'   # Ensure eval is off

# Data collator - Let it handle label creation and padding
# It will pad input_ids and create corresponding attention_mask
# It will shift input_ids to create labels, masking padding tokens with -100
print('Initializing Data Collator...')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print('Setting up training arguments...')
# Calculate warmup steps based on ratio if specified
if (
    config_values.get('warmup_ratio') is not None
    and config_values['eval_strategy'] != 'no'
):
    # Estimate total training steps
    # This estimation might be slightly off if the last batch is smaller
    total_train_batch_size = (
        config_values['per_device_train_batch_size']
        * accelerator.num_processes
        * config_values[  # Account for distributed training
            'gradient_accumulation_steps'
        ]
    )
    # Estimate steps per epoch
    steps_per_epoch = math.ceil(len(train_dataset) / total_train_batch_size)
    # Total estimated steps
    total_steps = int(steps_per_epoch * config_values['num_train_epochs'])
    warmup_steps = int(total_steps * config_values['warmup_ratio'])
    print(
        f"Calculated warmup steps: {warmup_steps} ({config_values['warmup_ratio']*100:.1f}% of estimated {total_steps} total steps)"
    )
else:
    # Fallback or if ratio isn't used
    warmup_steps = config_values.get(
        'warmup_steps', 0
    )   # Use fixed steps if provided, else 0
    print(f'Using fixed warmup steps: {warmup_steps}')


training_args = TrainingArguments(
    output_dir=config_values['output_dir'],
    overwrite_output_dir=config_values['overwrite_output_dir'],
    num_train_epochs=config_values['num_train_epochs'],
    per_device_train_batch_size=config_values['per_device_train_batch_size'],
    gradient_accumulation_steps=config_values['gradient_accumulation_steps'],
    gradient_checkpointing=config_values['gradient_checkpointing'],
    # Evaluation args
    # Ensure eval_strategy is 'no' if eval_dataset is None
    eval_strategy=config_values['eval_strategy']
    if eval_dataset is not None
    else 'no',
    eval_steps=config_values['eval_steps']
    if eval_dataset is not None and config_values['eval_strategy'] != 'no'
    else None,
    # Save args
    save_strategy=config_values[
        'eval_strategy'
    ],  # Often align save strategy with eval
    save_steps=config_values['save_steps'],
    save_total_limit=config_values['save_total_limit'],
    load_best_model_at_end=True
    if eval_dataset is not None and config_values['eval_strategy'] != 'no'
    else False,  # Load best model based on eval
    metric_for_best_model='loss'
    if eval_dataset is not None and config_values['eval_strategy'] != 'no'
    else None,
    greater_is_better=False
    if eval_dataset is not None and config_values['eval_strategy'] != 'no'
    else None,  # For loss, lower is better
    # Logging args
    logging_strategy='steps',
    logging_steps=config_values['logging_steps'],
    report_to=config_values['report_to'],
    # Optimizer args
    learning_rate=config_values['learning_rate'],
    weight_decay=config_values['weight_decay'],
    adam_beta1=config_values['adam_beta1'],
    adam_beta2=config_values['adam_beta2'],
    adam_epsilon=config_values['adam_epsilon'],
    lr_scheduler_type=config_values['lr_scheduler_type'],
    warmup_steps=warmup_steps,  # Use calculated or fixed warmup steps
    max_grad_norm=config_values['max_grad_norm'],
    # Other args
    dataloader_num_workers=config_values['dataloader_num_workers'],
    bf16=config_values['bf16'],
    seed=config_values['seed'],
    logging_first_step=True,  # Log metrics at the first step
    # Use accelerator device placement map automatically via Trainer
    torch_compile=True,  # Experimental: requires PyTorch 2.0+, can speed up training
)

print('Initializing Trainer...')
trainer = Trainer(
    model=model,  # Model already moved to device if accelerator was used earlier
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass validation dataset here (can be None)
    tokenizer=tokenizer,
    data_collator=data_collator,  # Use the LM collator
)

print('Starting training...')
try:
    train_result = (
        trainer.train()
    )   # Potentially resume_from_checkpoint=True/path argument here

    print(
        'Training finished. Saving final model (best checkpoint if evaluated)...'
    )
    # Save model using trainer's method (handles FSDP/DDP saving)
    # If load_best_model_at_end=True, this saves the best one found during training.
    # Otherwise, it saves the model from the final training state.
    trainer.save_model()
    trainer.save_state()   # Save optimizer, scheduler, rng states etc.

    # Log final training metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)   # Add sample count
    trainer.log_metrics(
        'train_end', metrics
    )   # Use different key for final train metrics
    trainer.save_metrics('train', metrics)   # Save final train metrics

    # Evaluate at the end and log/save metrics
    if eval_dataset is not None and training_args.eval_strategy != 'no':
        print('\n*** Evaluate Final Model ***')
        eval_metrics = (
            trainer.evaluate()
        )   # Evaluates the currently loaded model (best or final)
        eval_metrics['eval_samples'] = len(eval_dataset)
        try:
            # Ensure 'eval_loss' key exists
            if 'eval_loss' in eval_metrics:
                perplexity = math.exp(eval_metrics['eval_loss'])
            else:
                perplexity = float('inf')
                print("Warning: 'eval_loss' not found in evaluation metrics.")
        except OverflowError:
            perplexity = float('inf')
        except TypeError:   # Handle if eval_loss is None
            perplexity = float('inf')
            print(
                "Warning: 'eval_loss' was None during perplexity calculation."
            )

        eval_metrics['perplexity'] = perplexity
        trainer.log_metrics(
            'eval_end', eval_metrics
        )   # Use different key for final eval metrics
        trainer.save_metrics('eval', eval_metrics)   # Save final eval metrics

    print(f"\nTraining complete. Model saved to {config_values['output_dir']}")

except Exception as train_error:
    print(f'\nAn error occurred during training: {train_error}')
    import traceback

    traceback.print_exc()

finally:
    # Ensure wandb run finishes
    wandb.finish()
    print('Wandb run finished.')
