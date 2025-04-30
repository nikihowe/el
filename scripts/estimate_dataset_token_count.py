import argparse
import json
import os

from tqdm import tqdm
from transformers import GPT2TokenizerFast


def count_abstract_tokens(
    filename, field_name='abstract', approximation_percent=10
):
    """
    Reads the first N percent (default: 10%) of a large JSON Lines (JSONL) file
    by byte size, shows a progress bar, tokenizes the text from a specified
    field, counts tokens/abstracts, and extrapolates to estimate the total.

    Args:
        filename (str): The path to the JSON Lines file.
        field_name (str): The name of the field containing the text to tokenize.
        approximation_percent (int): The percentage of the file to process (1-100).
    """
    if not 1 <= approximation_percent <= 100:
        print('Error: Approximation percentage must be between 1 and 100.')
        return

    print(f'Initializing tokenizer...')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    print(
        f"Estimating token count for field '{field_name}' in file: {filename}"
    )
    print(f'Processing first {approximation_percent}% of the file...')

    total_tokens = 0
    line_count = 0
    processed_abstracts_count = 0
    bytes_processed = 0
    target_bytes = None

    errors = {
        'json_decode': 0,
        'missing_field': 0,
        'non_string': 0,
    }

    try:
        # Get file size to calculate target bytes
        try:
            file_size = os.path.getsize(filename)
            if file_size == 0:
                print('File is empty. No tokens to count.')
                return
            target_bytes = int(file_size * (approximation_percent / 100))
            # Ensure target_bytes is at least 1 if file is very small but not empty
            target_bytes = max(target_bytes, 1)
            pbar_total = target_bytes
            pbar_unit = 'B'
        except OSError as e:
            print(f'Error getting file size: {e}')
            print('Cannot perform approximation based on file size.')
            return   # Cannot proceed without file size for byte-based approximation

        with open(filename, 'r', encoding='utf-8') as f, tqdm(
            total=pbar_total,
            unit=pbar_unit,
            unit_scale=True,
            desc=f'Processing first {approximation_percent}%',
        ) as pbar:

            for line in f:
                line_bytes = len(line.encode('utf-8'))
                # Check if processing this line would exceed the target
                if (
                    bytes_processed + line_bytes > target_bytes
                    and line_count > 0
                ):
                    # Update progress bar to the target to show completion
                    pbar.update(target_bytes - bytes_processed)
                    break   # Stop processing

                bytes_processed += line_bytes
                pbar.update(line_bytes)
                line_count += 1

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        abstract_text = data.get(field_name)
                        if abstract_text is not None:
                            if isinstance(abstract_text, str):
                                tokens = tokenizer(
                                    abstract_text, add_special_tokens=True
                                )
                                total_tokens += len(tokens['input_ids'])
                                processed_abstracts_count += 1
                            else:
                                errors['non_string'] += 1
                        else:
                            errors['missing_field'] += 1
                    else:
                        errors['json_decode'] += 1
                except json.JSONDecodeError:
                    errors['json_decode'] += 1
                except Exception as e:
                    print(f'\nUnexpected error on line {line_count}: {e}')

        print(
            f'\nFinished processing {line_count} lines ({bytes_processed / (1024*1024):.2f} MB, approx. {approximation_percent}% of file).'
        )

        # Extrapolate results
        if processed_abstracts_count > 0 and bytes_processed > 0:
            # More accurate scaling factor based on actual bytes processed vs target
            scale_factor = file_size / bytes_processed
            # Simpler scaling by percentage target
            # scale_factor = 100 / approximation_percent
            estimated_total_abstracts = int(
                processed_abstracts_count * scale_factor
            )
            estimated_total_tokens = int(total_tokens * scale_factor)
        else:
            estimated_total_abstracts = 0
            estimated_total_tokens = 0

    except FileNotFoundError:
        print(f'Error: File not found at {filename}')
        return
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        return

    print(
        f'\n--- Tokenization Estimate (based on first {approximation_percent}%) ---'
    )
    print(f'Abstracts processed in sample: {processed_abstracts_count}')
    print(f'Tokens counted in sample: {total_tokens}')
    print(f'Estimated total abstracts in file: ~{estimated_total_abstracts:,}')
    print(
        f"Estimated total tokens for '{field_name}' in file: ~{estimated_total_tokens:,}"
    )

    # Report errors found in the sample
    if any(errors.values()):
        print('\nEncountered issues in the sample:')
        if errors['json_decode'] > 0:
            print(
                f"- Lines/Objects with JSON decoding errors: {errors['json_decode']}"
            )
        if errors['missing_field'] > 0:
            print(
                f"- Objects missing the '{field_name}' field: {errors['missing_field']}"
            )
        if errors['non_string'] > 0:
            print(
                f"- Objects where '{field_name}' was not a string: {errors['non_string']}"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f'Estimate token count in a specific field of a JSONL file by processing the first N%.'
    )
    parser.add_argument('json_file', help='Path to the large JSON Lines file.')
    parser.add_argument(
        '--field',
        default='abstract',
        help='Name of the field containing text to tokenize (default: abstract).',
    )
    parser.add_argument(
        '--percent',
        type=int,
        default=10,
        help='Percentage of file (by size) to process for estimation (default: 10).',
    )

    args = parser.parse_args()

    count_abstract_tokens(
        args.json_file,
        field_name=args.field,
        approximation_percent=args.percent,
    )
