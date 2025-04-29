import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_samples(model_path, num_samples=3, max_length=100):
    """Loads the model and tokenizer, then generates text samples."""

    print(f'Loading model from {model_path}...')
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure pad_token is set if it's missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print('Set pad_token to eos_token')

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency if supported
    )

    # Set device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()   # Set model to evaluation mode

    print(f'Model loaded on {device}.')

    # Sample prompts (feel free to change these)
    prompts = [
        'The study focuses on',
        'Abstract: We propose a novel method for',
        'Introduction\n\nRecent advances in machine learning',
    ]

    if len(prompts) > num_samples:
        prompts = prompts[:num_samples]
    elif len(prompts) < num_samples:
        # Add more generic prompts if needed
        prompts.extend(
            [''] * (num_samples - len(prompts))
        )   # Start with empty prompts for variety

    print(
        f'\n--- Generating {num_samples} samples (max_length={max_length}) ---'
    )
    for i, prompt_text in enumerate(prompts):
        print(f"\nSample {i+1}: Prompt: '{prompt_text}'")

        inputs = tokenizer(
            prompt_text, return_tensors='pt', padding=True, truncation=True
        ).to(device)

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,  # Enable sampling
                top_k=50,  # Consider top K tokens
                top_p=0.95,  # Use nucleus sampling
                temperature=0.7,  # Control randomness
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print('\nGenerated Text:')
        print(generated_text)
        print('-' * 20)


if __name__ == '__main__':
    model_directory = './pythia-70m-arxiv-scratch/'
    generate_samples(model_directory)
