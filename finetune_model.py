import json
import os

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb
from datasets import Dataset, DatasetDict

from constants import BASE_MODEL_DIR, FINETUNED_MODEL_DIR, FINETUNED_DATA_PATH

LOG_TO_WANDB = True
if LOG_TO_WANDB:
    wandb.init(project='el_takehome')

# Paths
TRAIN_FILE = os.path.join(FINETUNED_DATA_PATH, 'train.jsonl')
DEV_FILE = os.path.join(FINETUNED_DATA_PATH, 'dev.jsonl')

# 1. Load tokenizer
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load and preprocess datasets
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_label_list(data):
    return sorted(list(set(ex['label'] for ex in data)))

print('Loading datasets...')
train_data = load_jsonl(TRAIN_FILE)
dev_data = load_jsonl(DEV_FILE)
labels = get_label_list(train_data + dev_data)
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# Convert to HuggingFace Dataset
train_ds = Dataset.from_list(
    [{'text': ex['text'], 'label': label2id[ex['label']]} for ex in train_data]
)
dev_ds = Dataset.from_list(
    [{'text': ex['text'], 'label': label2id[ex['label']]} for ex in dev_data]
)
datasets = DatasetDict({'train': train_ds, 'validation': dev_ds})

# 3. Tokenize
def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding=False)

datasets = datasets.map(preprocess, batched=True)

# 4. Compute class weights (no sklearn)
train_label_indices = [label2id[l] for l in [ex['label'] for ex in train_data]]
num_classes = len(labels)
counts = np.bincount(train_label_indices, minlength=num_classes)
total = sum(counts)
class_weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f'Class weights: {class_weights}')

# 5. Load base model with new classification head
print('Loading base model with new classification head...')
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_DIR,
    num_labels=len(labels),
    problem_type='single_label_classification',
    ignore_mismatched_sizes=True,  # allow new head
    id2label=id2label,
    label2id=label2id,
)

# 6. Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# 7. Metrics
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted'),
    }

# 8. Custom Trainer to use weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(model.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 9. Training arguments
training_args = TrainingArguments(
    output_dir=FINETUNED_MODEL_DIR,
    eval_strategy='steps',
    eval_steps=10,
    save_strategy='steps',
    save_steps=10,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir=os.path.join(FINETUNED_MODEL_DIR, 'logs'),
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_total_limit=1,
    report_to=['wandb'],
)

# 10. Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['validation'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 11. Train
print('Starting finetuning...')
trainer.train()

# 12. Save final model
print(f'Saving finetuned model to {FINETUNED_MODEL_DIR}')
trainer.save_model(FINETUNED_MODEL_DIR)
tokenizer.save_pretrained(FINETUNED_MODEL_DIR)
print('Done.')

if LOG_TO_WANDB:
    wandb.finish()
    print('Wandb run finished.')
