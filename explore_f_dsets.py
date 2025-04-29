import json
import os
from collections import Counter, defaultdict

DATASET_DIR = 'finetuning_datasets'
FILES = ['train.jsonl', 'dev.jsonl', 'test_no_labels.jsonl']


def detect_label_key(sample):
    """Try to guess the label key from a sample dict."""
    for key in sample:
        if key.lower() == 'label':
            return key
    # fallback: look for a string value that isn't 'text' or 'meta'
    for key in sample:
        if key.lower() not in {'text', 'meta'} and isinstance(
            sample[key], str
        ):
            return key
    return None


def label_distribution(filepath):
    with open(filepath, 'r') as f:
        first = f.readline()
        if not first:
            print(f'{filepath}: File is empty.')
            return None, None
        sample = json.loads(first)
        label_key = detect_label_key(sample)
        if not label_key:
            print(f'{filepath}: No label key found.')
            return None, None
        counter = Counter()
        counter[sample[label_key]] += 1
        for line in f:
            obj = json.loads(line)
            counter[obj[label_key]] += 1
        return label_key, counter


def main():
    for fname in FILES:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            print(f'{fname}: File not found.')
            continue
        print(f'\n=== {fname} ===')
        label_key, counter = label_distribution(path)
        if counter is None:
            print('No label distribution to show.')
            continue
        total = sum(counter.values())
        print(f'Label key: {label_key}')
        for label, count in counter.most_common():
            print(f'  {label}: {count} ({count/total:.2%})')
        print(f'Total examples: {total}')


if __name__ == '__main__':
    main()
