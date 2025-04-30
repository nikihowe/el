import os
import sys

from datasets import DatasetDict, load_dataset, load_from_disk

# Original dataset
DATA_PATH = 'datasets/pretraining/arxiv-metadata-oai-snapshot.jsonl'
# Tokenized but not grouped
TOKENIZED_DATA_PATH = 'datasets/pretraining/tokenized_datasets'
# Finished processed dataset, ready for training
PROCESSED_DATA_PATH = 'datasets/pretraining/grouped_datasets'


def tokenize_function(examples, config_values, tokenizer):
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
    )


def group_texts(examples, block_size):
    """Concatenates texts from a batch and chunks them into blocks of fixed size."""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        print(
            f'Dropping {total_length - block_size} tokens out of {total_length} total tokens'
        )

    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    return result


def load_datasets(config_values, tokenizer, DEBUG):
    # --- Data Loading and Processing ---
    print('\n--- Data Loading and Processing ---')
    lm_datasets = None
    tokenized_datasets = None
    split_dataset = None

    # 1. Try load final grouped data
    try:
        print(
            f'Attempting to load saved grouped datasets from {PROCESSED_DATA_PATH}...'
        )
        lm_datasets = load_from_disk(PROCESSED_DATA_PATH)
        print('Successfully loaded saved grouped datasets!')
        # Optional: Inspect loaded data
        if DEBUG and lm_datasets:
            print('\nInspecting loaded grouped data examples:')
            num_processed_to_show = min(3, len(lm_datasets['train']))
            for i in range(num_processed_to_show):
                example = lm_datasets['train'][i]
                print(f'\nExample {i}:')
                print(f"Input IDs length: {len(example['input_ids'])}")
                print(
                    f"First 50 tokens: {tokenizer.decode(example['input_ids'][:50])}"
                )
    except FileNotFoundError:
        print(f'Grouped dataset not found at {PROCESSED_DATA_PATH}.')
    except Exception as e:
        print(
            f'Could not load grouped dataset due to error: {e}. Will try to regenerate.'
        )

    # 2. If grouped failed, try load tokenized data
    if lm_datasets is None:
        try:
            if not config_values['overwrite_cache']:
                print(
                    f'Attempting to load tokenized datasets from {TOKENIZED_DATA_PATH}...'
                )
                tokenized_datasets = load_from_disk(TOKENIZED_DATA_PATH)
                print('Successfully loaded cached tokenized datasets.')
            else:
                print(
                    "'overwrite_cache' is True, skipping load of tokenized data."
                )
                raise FileNotFoundError   # Force regeneration if overwrite is True
        except FileNotFoundError:
            print(
                f'Tokenized dataset not found at {TOKENIZED_DATA_PATH} or overwrite requested.'
            )
        except Exception as e:
            print(
                f'Could not load tokenized dataset due to error: {e}. Will try to regenerate.'
            )

    # 3. If tokenized failed, load and split raw data
    if tokenized_datasets is None and lm_datasets is None:
        print(f"Loading raw dataset: {DATA_PATH}...")
        if not os.path.exists(DATA_PATH):
            print(
                f"Error: Raw dataset file not found at {DATA_PATH}"
            )
            sys.exit(1)
        try:
            raw_dataset = load_dataset(
                'json', data_files=DATA_PATH, split='train'
            )
            raw_datasets = DatasetDict({'train': raw_dataset})

            if DEBUG:
                print('\nInspecting raw data examples:')
                num_examples_to_show = min(3, len(raw_datasets['train']))
                for i in range(num_examples_to_show):
                    example = raw_datasets['train'][i]
                    abstract = example.get(config_values['text_field'], 'N/A')
                    print(f'\nExample {i}:')
                    print(f'Abstract: {abstract[:200]}...')
                    print(f'Length: {len(abstract)}')

            print('Splitting raw dataset...')
            if config_values['validation_split_percentage'] > 0:
                split_dataset = raw_datasets['train'].train_test_split(
                    test_size=config_values['validation_split_percentage']
                    / 100.0,
                    seed=config_values['seed'],
                )
                split_dataset['validation'] = split_dataset.pop('test')
                print(
                    f"Split dataset into {100-config_values['validation_split_percentage']}% train and {config_values['validation_split_percentage']}% validation."
                )
            else:
                print(
                    'Using entire dataset for training (no validation split).'
                )
                split_dataset = DatasetDict({'train': raw_datasets['train']})
                # config_values['eval_strategy'] = 'no' # Trainer args handle this later

        except Exception as load_e:
            print(f'Error loading or splitting raw dataset: {load_e}')
            sys.exit(1)

    # 4. If tokenized data is missing but raw exists, tokenize
    if (
        tokenized_datasets is None
        and split_dataset is not None
        and lm_datasets is None
    ):
        print('Applying tokenization...')
        try:
            tokenized_datasets = split_dataset.map(
                lambda examples: tokenize_function(
                    examples, config_values, tokenizer
                ),
                batched=True,
                batch_size=config_values['map_batch_size'],
                num_proc=config_values['preprocessing_num_workers'],
                remove_columns=next(iter(split_dataset.values())).column_names,
                load_from_cache_file=False,  # Cache handled by checking TOKENIZED_DATA_PATH
                desc='Running tokenizer on dataset splits',
                writer_batch_size=config_values['map_batch_size'],
            )
            save_dir = os.path.dirname(TOKENIZED_DATA_PATH)
            os.makedirs(save_dir, exist_ok=True)
            print(f'Saving tokenized datasets to {TOKENIZED_DATA_PATH}...')
            tokenized_datasets.save_to_disk(TOKENIZED_DATA_PATH)
        except Exception as tokenize_e:
            print(f'Error during tokenization: {tokenize_e}')
            sys.exit(1)

    # 5. If grouped data is missing but tokenized exists, group
    if lm_datasets is None and tokenized_datasets is not None:
        # Verify splits before grouping if loaded from cache
        if split_dataset and set(tokenized_datasets.keys()) != set(
            split_dataset.keys()
        ):
            print(
                'Warning: Loaded tokenized dataset splits do not match current splits defined by validation_split_percentage. Re-tokenizing might be needed if config changed.'
            )
            # Decide if this should be an error or just a warning

        block_size = config_values['block_size']
        print(f'Grouping texts into blocks of size {block_size}...')
        try:
            lm_datasets = tokenized_datasets.map(
                lambda examples: group_texts(examples, block_size),
                batched=True,
                batch_size=config_values['map_batch_size'],
                num_proc=config_values['preprocessing_num_workers'],
                load_from_cache_file=False,  # Cache handled by checking PROCESSED_DATA_PATH
                desc=f'Grouping texts into chunks of {block_size}',
                writer_batch_size=config_values['map_batch_size'],
                # Keep only columns returned by group_texts (input_ids)
                remove_columns=[
                    col
                    for col in tokenized_datasets['train'].column_names
                    if col != 'input_ids'
                ],
            )

            # Filter out empty examples potentially created by group_texts
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
            save_dir = os.path.dirname(PROCESSED_DATA_PATH)
            os.makedirs(save_dir, exist_ok=True)
            print(f'Saving grouped datasets to {PROCESSED_DATA_PATH}...')
            lm_datasets.save_to_disk(PROCESSED_DATA_PATH)
        except Exception as group_e:
            print(f'Error during grouping: {group_e}')
            sys.exit(1)
