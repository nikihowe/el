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

DATASET_PATH = 'datasets/arxiv-metadata-oai-snapshot.jsonl'
LOG_TO_WANDB = False

# Set cache directory to scratch
# NOTE: This is necessary because the default HF cache writes to $HOME,
# which has a storage cap of 100GB and will fill up.
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ[
    'TOKENIZERS_PARALLELISM'
] = 'false'   # Often helps avoid deadlocks with num_workers > 0

# --- Wandb Logging Setup ---
wandb.init(project='el_takehome')

# --- Hardcoded Configuration ---
config_values = {
    # Model & Tokenizer
    'model_name': 'EleutherAI/pythia-70m',
    'block_size': 2048,
    # Data
    'dataset_path': DATASET_PATH,
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
    'fp16': False,  # NOTE: fp16 caused `Attempting to unscale FP16 gradients` error
    'bf16': True,
    'fp16_full_eval': False,
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

# Basic check for dataset file existence
if not os.path.exists(config_values['dataset_path']):
    print(f"Error: Dataset file not found at {config_values['dataset_path']}")
    sys.exit(1)

# Check CUDA availability for fp16
if config_values['fp16'] and not torch.cuda.is_available():
    print(
        'Warning: fp16 is enabled in config, but CUDA is not available. Training will proceed without fp16.'
    )
    config_values['fp16'] = False   # Disable fp16 if CUDA not found

# Initialize Accelerator for device placement and fp16 handling
# Determine mixed precision mode for Accelerator based on config AND support checks
if config_values[
    'bf16'
]:   # bf16 takes precedence if enabled and supported (checked above)
    precision_mode = 'bf16'
elif config_values['fp16']:   # Check fp16 only if bf16 is not used
    precision_mode = 'fp16'
else:
    precision_mode = 'no'   # Default to no mixed precision

print(f"Initializing Accelerator with mixed_precision='{precision_mode}'")
accelerator = Accelerator(mixed_precision=precision_mode)
print(f'Using device: {accelerator.device}')

print('Loading tokenizer...')
# Explicitly add pad token if missing - common for GPT-like models
tokenizer = AutoTokenizer.from_pretrained(config_values['model_name'])
if tokenizer.pad_token is None:
    print(
        'Tokenizer does not have a pad token. Adding eos_token as pad_token.'
    )
    tokenizer.pad_token = tokenizer.eos_token
# Set padding side appropriately for Causal LM if needed (though collator often handles)
# tokenizer.padding_side = 'left' # Or 'right' depending on model/preference

print(f"Loading configuration for {config_values['model_name']}...")
model_config = AutoConfig.from_pretrained(
    config_values['model_name'],
    trust_remote_code=True,
    use_cache=False,  # Required for gradient checkpointing
    # Ensure vocab size matches tokenizer, resize_token_embeddings will handle mismatch later
    # vocab_size=len(tokenizer), # Usually handled by from_pretrained or resize
    pad_token_id=tokenizer.pad_token_id,  # Ensure config knows pad token id
)

# Ensure the model is configured for causal language modeling - usually automatic for AutoModelForCausalLM
# model_config.is_decoder = True
# model_config.add_cross_attention = False # Causal LM doesn't use cross-attention

print(f"Initializing {config_values['model_name']} model from scratch...")
# from_config without pretrained weights initializes the model with random weights
model = AutoModelForCausalLM.from_config(model_config)
model.config.use_cache = (
    False  # Explicitly disable caching for gradient checkpointing again
)

# Apply GPT-J initialization (Optional, but can be better than default random)
print('\nApplying GPT-J initialization...')


def gptj_init(module):
    if isinstance(module, (torch.nn.Linear,)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)


model.apply(gptj_init)


# Resize token embeddings in case tokenizer vocab size differs from config (e.g., added pad token)
model.resize_token_embeddings(len(tokenizer))
print(f'Resized model token embeddings to: {len(tokenizer)}')

# --- Test a single forward pass (Optional but recommended) ---
print('\nTesting single forward pass...')
# Move model to the correct device *before* testing
model = model.to(accelerator.device)   # Use accelerator's device
print(f'Model device: {next(model.parameters()).device}')

test_input_text = 'This is a test.'
test_input = tokenizer(test_input_text, return_tensors='pt')
print(f"Input device before moving: {test_input['input_ids'].device}")
# Move input to the correct device
test_input = {k: v.to(accelerator.device) for k, v in test_input.items()}
print(f"Input device after moving: {test_input['input_ids'].device}")

# Add labels for loss calculation (shifted input_ids)
test_input['labels'] = test_input[
    'input_ids'
].clone()   # For Causal LM, labels are usually shifted input_ids
print(f'Test input keys: {test_input.keys()}')
print(f"Test input shape: {test_input['input_ids'].shape}")
print(f"Test labels shape: {test_input['labels'].shape}")


model.eval()   # Set model to evaluation mode for the test pass
with torch.no_grad():
    try:
        outputs = model(**test_input)
        print(f'Outputs type: {type(outputs)}')
        # print(f"Outputs attributes: {dir(outputs)}") # Can be verbose
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            print(f'Test forward pass loss: {outputs.loss.item():.6f}')
        else:
            print('Loss attribute not found or is None in outputs.')
            # Try to compute loss manually if needed for debugging
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = test_input['labels'][..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits_flat = shift_logits.view(
                    -1, shift_logits.size(-1)
                )
                shift_labels_flat = shift_labels.view(-1)
                # Ensure labels are in the valid range [0, vocab_size-1] and not -100
                active_loss = (
                    shift_labels_flat != -100
                )   # Should not be -100 here as we didn't use collator
                if active_loss.any():
                    loss = loss_fct(
                        shift_logits_flat[active_loss],
                        shift_labels_flat[active_loss],
                    )
                    print(f'Manually computed loss: {loss.item():.6f}')
                else:
                    print(
                        'No active labels for manual loss calculation (all labels might be padding).'
                    )

            else:
                print('No logits found in outputs either.')
    except Exception as e:
        print(f'Error during test forward pass: {e}')
        import traceback

        traceback.print_exc()
model.train()   # Set model back to training mode

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
    # Tokenize valid texts. Padding handled later by collator. Truncation might be needed if texts are very long.
    # return tokenizer(valid_texts, truncation=False) # Let group_texts handle chunking
    # Or, if individual abstracts can exceed block_size significantly *before* concatenation:
    return tokenizer(
        valid_texts,
        truncation=True,
        max_length=config_values['block_size'] * 2,
        padding=False,
    )   # Added truncation safeguard


# --- Block Processing Function ---
block_size = config_values['block_size']


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
        # Handle cases where total concatenated length is less than block size
        # Option 1: Skip this batch (return empty dict)
        # return {}
        # Option 2: Pad (requires collator to handle padding correctly)
        # Keep as is for now, Trainer might handle small final batch if not dropped
        pass

    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # ***** THE FIX: DO NOT CREATE LABELS HERE *****
    # Let the DataCollatorForLanguageModeling handle label creation (shifting)
    # result["labels"] = result["input_ids"].copy() # <- REMOVED THIS LINE
    return result


# --- End Block Processing Function ---


print(f"Loading dataset: {config_values['dataset_path']}...")
# Load the JSONL dataset (expecting a 'train' split by default)
# Consider streaming if dataset is huge: streaming=True
try:
    raw_dataset = load_dataset(
        'json', data_files=config_values['dataset_path'], split='train'
    )   # Explicitly request train split
    raw_datasets = DatasetDict(
        {'train': raw_dataset}
    )   # Put it in a DatasetDict
except Exception as e:
    print(f'Error loading dataset: {e}')
    print(
        "Please ensure the jsonl file is correctly formatted and contains a 'train' structure or adjust split name."
    )
    sys.exit(1)


# Inspect raw data
print('\nInspecting raw data examples:')
num_examples_to_show = min(3, len(raw_datasets['train']))
for i in range(num_examples_to_show):
    example = raw_datasets['train'][i]
    abstract = example.get(
        config_values['text_field'], 'N/A'
    )   # Safely get abstract
    print(f'\nExample {i}:')
    print(f'Abstract: {abstract[:200]}...')  # First 200 chars
    print(f'Length: {len(abstract)}')

# Try to load saved processed datasets
processed_path = 'processed_datasets'
tokenized_path = 'tokenized_datasets'   # Added path for tokenized cache
try:
    print(
        f'\nAttempting to load saved processed datasets from {processed_path}...'
    )
    if os.path.exists(processed_path):
        lm_datasets = load_from_disk(processed_path)
        print('Successfully loaded saved processed datasets!')

        # Verify expected columns are present
        if 'input_ids' not in lm_datasets['train'].column_names:
            raise ValueError("Loaded dataset missing 'input_ids' column.")
        # DO NOT expect 'labels' here anymore after the fix
        # if "labels" not in lm_datasets["train"].column_names:
        #      raise ValueError("Loaded dataset missing 'labels' column.")

        print('\nInspecting processed data examples:')
        num_processed_to_show = min(3, len(lm_datasets['train']))
        for i in range(num_processed_to_show):
            example = lm_datasets['train'][i]
            print(f'\nExample {i}:')
            print(f"Input IDs length: {len(example['input_ids'])}")
            print(
                f"First 50 tokens: {tokenizer.decode(example['input_ids'][:50])}"
            )
            # print(f"Labels length: {len(example['labels'])}") # Labels no longer present here
            # print(f"First 50 labels: {tokenizer.decode(example['labels'][:50])}")
    else:
        raise FileNotFoundError(f"Directory '{processed_path}' not found")

except Exception as e:
    print(
        f'Could not load saved processed datasets: {e}. Processing from scratch...'
    )

    # Split the dataset into training and validation
    if config_values['validation_split_percentage'] > 0:
        split_dataset = raw_datasets['train'].train_test_split(
            test_size=config_values['validation_split_percentage'] / 100.0,
            seed=config_values['seed'],  # Use seed for reproducibility
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
        ] = 'no'   # Disable eval if no validation set
        print('Evaluation disabled as no validation split was created.')

    print('Applying tokenization...')
    # Try loading tokenized cache first
    try:
        if not config_values['overwrite_cache'] and os.path.exists(
            tokenized_path
        ):
            print(
                f'Attempting to load tokenized datasets from {tokenized_path}...'
            )
            tokenized_datasets = load_from_disk(tokenized_path)
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
        print(f'Could not load tokenized cache: {te}. Tokenizing...')
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
        print(f'Saving tokenized datasets to {tokenized_path}...')
        tokenized_datasets.save_to_disk(tokenized_path)

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
        # remove_columns=tokenized_datasets["train"].column_names # Keep only the columns returned by group_texts (input_ids)
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

    print(f'Saving processed datasets to {processed_path}...')
    lm_datasets.save_to_disk(processed_path)

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
    # This estimation might be slightly off if the last batch is smaller, but it's usually close enough
    # Ensure train_dataset is loaded before this step
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
    fp16=config_values['fp16'],
    bf16=config_values['bf16'],
    # fp16_opt_level=config_values.get("fp16_opt_level"), # Usually handled by Accelerator/Trainer
    seed=config_values['seed'],
    logging_first_step=True,  # Log metrics at the first step
    # Use accelerator device placement map automatically via Trainer
    # ddp_find_unused_parameters=False, # Set if encountering DDP issues, usually not needed
    # torch_compile=True, # Experimental: requires PyTorch 2.0+, can speed up training
)

print('Initializing Trainer...')
trainer = Trainer(
    model=model,  # Model already moved to device if accelerator was used earlier
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass validation dataset here (can be None)
    tokenizer=tokenizer,
    data_collator=data_collator,  # Use the LM collator
    # compute_metrics=compute_metrics, # Optional: if you want more than just loss/perplexity
    # optimizers = (optimizer, scheduler) # Optional: provide custom optimizer/scheduler
)

# Move model to accelerator device (redundant if done before, but safe)
# model = accelerator.prepare(model) # Trainer usually handles this with args

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
    # Optionally save state even on error
    # print("Attempting to save state on error...")
    # trainer.save_model(os.path.join(config_values["output_dir"], "error_checkpoint"))
    # trainer.save_state()

finally:
    # Ensure wandb run finishes
    wandb.finish()
    print('Wandb run finished.')
