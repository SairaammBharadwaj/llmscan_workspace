import torch
import numpy as np


def scan_hidden_states_batch(prompts, model, tokenizer, device):
    print("\n--- Batch Hidden-State Scan ---")
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    batch_size = hidden_states[0].shape[0]
    num_layers = len(hidden_states)
    batch_results = []
    for batch_idx in range(batch_size):
        transition_scores = []
        activation_stats = []
        for layer_idx in range(num_layers - 1):
            current_layer = hidden_states[layer_idx][batch_idx]
            next_layer = hidden_states[layer_idx + 1][batch_idx]
            transition = torch.norm(next_layer - current_layer, p=2).item()
            transition_scores.append(transition)
            activation_stats.append(current_layer.abs().mean().item())
        activation_stats = np.array(activation_stats, dtype=np.float32)
        batch_results.append(
            {
                "transition_scores": transition_scores,
                "activation_stats": activation_stats.tolist(),
            }
        )
    return batch_results
