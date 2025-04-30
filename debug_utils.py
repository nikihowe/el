import traceback

import torch


def test_forward_pass(
    model, tokenizer, accelerator, test_input_text='This is a test.'
):
    """
    Performs a single forward pass with the model using a sample input
    to check for basic execution errors, device placement, and loss calculation.

    Args:
        model: The PyTorch model to test.
        tokenizer: The tokenizer associated with the model.
        accelerator: The Accelerate Accelerator object.
        test_input_text (str): The text to use for the test input.
    """
    print('\n--- Testing Single Forward Pass ---')
    original_training_state = model.training
    device = accelerator.device

    # Ensure model is on the correct device (should already be, but double-check)
    model = model.to(device)
    print(f'Model device: {next(model.parameters()).device}')

    # Prepare input
    test_input = tokenizer(test_input_text, return_tensors='pt')
    print(f"Input device before moving: {test_input['input_ids'].device}")
    test_input = {k: v.to(device) for k, v in test_input.items()}
    print(f"Input device after moving: {test_input['input_ids'].device}")

    # Add labels for Causal LM loss calculation
    test_input['labels'] = test_input['input_ids'].clone()
    print(f'Test input keys: {test_input.keys()}')
    print(f"Test input shape: {test_input['input_ids'].shape}")
    print(f"Test labels shape: {test_input['labels'].shape}")

    # Perform forward pass
    model.eval()  # Set model to evaluation mode for the test pass
    try:
        with torch.no_grad():
            outputs = model(**test_input)
            print(f'Outputs type: {type(outputs)}')

            if hasattr(outputs, 'loss') and outputs.loss is not None:
                print(f'Test forward pass loss: {outputs.loss.item():.6f}')
            else:
                print('Loss attribute not found or is None in outputs.')
                # Optional: Manual loss calculation for debugging
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = test_input['labels'][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    shift_logits_flat = shift_logits.view(
                        -1, shift_logits.size(-1)
                    )
                    shift_labels_flat = shift_labels.view(-1)
                    active_loss = shift_labels_flat != -100
                    if active_loss.any():
                        loss = loss_fct(
                            shift_logits_flat[active_loss],
                            shift_labels_flat[active_loss],
                        )
                        print(f'Manually computed loss: {loss.item():.6f}')
                    else:
                        print('No active labels for manual loss calculation.')
                else:
                    print('No logits found in outputs either.')

    except Exception as e:
        print(f'Error during test forward pass: {e}')
        traceback.print_exc()

    finally:
        # Restore original training state
        model.train(original_training_state)
        print('--- Finished Testing Single Forward Pass ---')
        if model.training != original_training_state:
            print(
                f'Warning: Model training state may not have been correctly restored. Expected: {original_training_state}, Got: {model.training}'
            )
