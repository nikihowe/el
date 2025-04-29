import os
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# Paths
FINETUNE_DIR = 'pythia-70m-arxiv-finetuned'
DATASET_DIR = 'finetuning_datasets'
DEV_FILE = os.path.join(DATASET_DIR, 'dev.jsonl')
TEST_FILE = os.path.join(DATASET_DIR, 'test_no_labels.jsonl')

# 1. Load fine-tuned model and tokenizer
print('Loading fine-tuned model and tokenizer...')
model = AutoModelForSequenceClassification.from_pretrained(FINETUNE_DIR)
tokenizer = AutoTokenizer.from_pretrained(FINETUNE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Ensure model is on the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Model loaded on {device}')

# Load label mappings from model config
id2label = model.config.id2label
label2id = model.config.label2id
labels = list(id2label.values())

# 2. Load and preprocess dev dataset for evaluation
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


print('Loading dev dataset...')
dev_data = load_jsonl(DEV_FILE)
dev_ds = Dataset.from_list(
    [{'text': ex['text'], 'label': label2id[ex['label']]} for ex in dev_data]
)


def preprocess(example):
    return tokenizer(example['text'], truncation=True, padding=False)


dev_ds = dev_ds.map(preprocess, batched=True)

# 3. Evaluate on Dev Set
print('Evaluating on dev set...')
data_collator = DataCollatorWithPadding(tokenizer)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

predictions = trainer.predict(dev_ds)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

# Calculate metrics
accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, preds, average='weighted'
)

print(f'Validation Metrics:')
print(f'  Accuracy: {accuracy:.4f}')
print(f'  F1 Score (Weighted): {f1:.4f}')
print(f'  Precision (Weighted): {precision:.4f}')
print(f'  Recall (Weighted): {recall:.4f}')

# 4. Load and preprocess test dataset for prediction examples
print('\nLoading test dataset for prediction examples...')
test_data = load_jsonl(TEST_FILE)
test_examples = test_data[:20]   # Take first 20 examples

# Convert to Dataset for easier processing
test_ds_subset = Dataset.from_list(
    [{'text': ex['text']} for ex in test_examples]  # No labels here
)

test_ds_subset = test_ds_subset.map(preprocess, batched=True)

# 5. Generate predictions for test examples
print('\nGenerating predictions for first 20 test examples...')
test_predictions = trainer.predict(test_ds_subset)
test_preds_indices = np.argmax(test_predictions.predictions, axis=-1)
test_preds_labels = [id2label[idx] for idx in test_preds_indices]

# 6. Display test predictions
print('\nTest Set Predictions (First 20 Examples):')
for i, example in enumerate(test_examples):
    print(f'--- Example {i+1} ---')
    print(f"Text: {example['text'][:200]}...")   # Show beginning of text
    print(f'Predicted Label: {test_preds_labels[i]}')
    print('-' * 20)

print('\nDone.')
