import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from datasets import Dataset
from constants import FINETUNED_MODEL_DIR, FINETUNED_DATA_PATH
from dataset_utils import check_token_lengths

# Paths
DEV_FILE = os.path.join(FINETUNED_DATA_PATH, 'dev.jsonl')
TEST_FILE = os.path.join(FINETUNED_DATA_PATH, 'test_no_labels.jsonl')
PREDICTIONS_DIR = 'predictions'
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, 'test_predictions.jsonl')

# Create predictions directory if it doesn't exist
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# 1. Load fine-tuned model and tokenizer
print('Loading fine-tuned model and tokenizer...')
model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get model's maximum block size from config
block_size = model.config.max_position_embeddings
print(f'Model maximum block size: {block_size}')

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
dev_texts = [ex['text'] for ex in dev_data]
check_token_lengths(dev_texts, tokenizer, block_size, 'dev set')

dev_ds = Dataset.from_list(
    [{'text': ex['text'], 'label': label2id[ex['label']]} for ex in dev_data]
)


def preprocess(example):
    return tokenizer(
        example['text'], truncation=True, padding=False, max_length=block_size
    )


dev_ds = dev_ds.map(preprocess, batched=True)

# 3. Evaluate on Dev Set
print('Evaluating on dev set...')
data_collator = DataCollatorWithPadding(tokenizer)
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
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
test_texts = [ex['text'] for ex in test_data]
check_token_lengths(test_texts, tokenizer, block_size, 'test set')

test_ds = Dataset.from_list(
    [{'text': ex['text']} for ex in test_data]  # No labels here
)
test_ds = test_ds.map(preprocess, batched=True)

# 5. Generate predictions for test examples
print('\nGenerating predictions for test set...')
test_predictions = trainer.predict(test_ds)
test_preds_indices = np.argmax(test_predictions.predictions, axis=-1)
test_preds_labels = [id2label[idx] for idx in test_preds_indices]

# Save predictions to file
print(f'\nSaving predictions to {PREDICTIONS_FILE}...')
with open(PREDICTIONS_FILE, 'w') as f:
    for example, pred_label in zip(test_data, test_preds_labels):
        prediction_entry = {
            'text': example['text'],
            'predicted_label': pred_label
        }
        f.write(json.dumps(prediction_entry) + '\n')

# 6. Display first 20 test predictions
print('\nTest Set Predictions (First 20 Examples):')
for i, (example, pred_label) in enumerate(zip(test_data[:20], test_preds_labels[:20])):
    print(f'--- Example {i+1} ---')
    print(f"Text: {example['text'][:200]}...")   # Show beginning of text
    print(f'Predicted Label: {pred_label}')
    print('-' * 20)

print('\nDone.')
