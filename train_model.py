import math
import os
import sys

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
from dataset_utils import load_datasets
from debug_utils import test_forward_pass

LOG_TO_WANDB = False
DEBUG = True
MODEL_SAVE_DIR = './models/pythia-70m-arxiv-scratch'

# Set cache directory to `scratch` partition
# NOTE: This is necessary because the default HF cache writes to $HOME,
# which has a storage cap of 100GB and will fill up.
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
# Avoids a warning about avoiding deadlocks with num_workers > 0
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if LOG_TO_WANDB:
    wandb.init(project='el_takehome')

config_values = {
    # Model & Tokenizer
    'model_name': 'EleutherAI/pythia-70m',
    'block_size': 2048,
    # Data
    'text_field': 'abstract',
    'validation_split_percentage': 5,  # Percentage of data to hold out for validation
    'preprocessing_num_workers': 8,  # Going too high caused memory issues
    'overwrite_cache': False,
    'dataloader_num_workers': 4,  # Going too high caused memory issues
    'map_batch_size': 1000,
    # Training
    'output_dir': MODEL_SAVE_DIR,
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
    'adam_beta2': 0.999,
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
# Initialize the model with random weights
model = AutoModelForCausalLM.from_config(model_config)

# Resize token embeddings in case tokenizer vocab size differs from config (e.g., added pad token)
model.resize_token_embeddings(len(tokenizer))
print(f'Resized model token embeddings to: {len(tokenizer)}')

if DEBUG:
    test_forward_pass(model, tokenizer, accelerator)

lm_datasets = load_datasets(config_values, tokenizer, DEBUG)
print('Splitting dataset into train and validation sets...')
train_dataset = lm_datasets['train']
eval_dataset = lm_datasets['validation']

print('Initializing Data Collator...')
# Causal language modeling, so mlm=False
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print('Calculating warmup steps...')
# Calculate the warmup steps
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

print('Setting up training arguments...')
training_args = TrainingArguments(
    output_dir=config_values['output_dir'],
    overwrite_output_dir=config_values['overwrite_output_dir'],
    num_train_epochs=config_values['num_train_epochs'],
    per_device_train_batch_size=config_values['per_device_train_batch_size'],
    gradient_accumulation_steps=config_values['gradient_accumulation_steps'],
    gradient_checkpointing=config_values['gradient_checkpointing'],
    # Evaluation args
    eval_strategy=config_values['eval_strategy'],
    eval_steps=config_values['eval_steps'],
    # Save args
    save_strategy=config_values['eval_strategy'],
    save_steps=config_values['save_steps'],
    save_total_limit=config_values['save_total_limit'],
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    greater_is_better=False,
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
    warmup_steps=warmup_steps,
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
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print('Starting training...')
try:
    train_result = trainer.train()

    print(
        'Training finished. Saving final model (best checkpoint if evaluated)...'
    )
    # Save model using trainer's method (handles FSDP/DDP saving)
    # Saves the best model found during training.
    trainer.save_model()
    trainer.save_state()

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
    if LOG_TO_WANDB:
        wandb.finish()
        print('Wandb run finished.')
