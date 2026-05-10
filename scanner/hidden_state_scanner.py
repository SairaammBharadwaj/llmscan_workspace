import torch
import numpy as np

def normalize_vector(x):

    norm = torch.norm(x, p=2)

    return x / (norm + 1e-8)

def extract_hidden_states(
    prompt,
    model,
    tokenizer,
    device
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states

    return hidden_states

def compute_layer_transitions(
    hidden_states
):

    layer_distances = []

    for i in range(
        len(hidden_states) - 1
    ):

        current_layer = hidden_states[i]

        next_layer = hidden_states[i + 1]

        current_layer = normalize_vector(
            current_layer.float()
        )

        next_layer = normalize_vector(
            next_layer.float()
        )

        distance = torch.norm(
            current_layer -
            next_layer,
            p=2
        ).item()

        layer_distances.append(distance)

    return layer_distances

def compute_activation_statistics(
    hidden_states
):

    statistics = []

    for layer in hidden_states:

        layer = layer.float()

        mean_val = torch.mean(layer).item()

        std_val = torch.std(layer).item()

        max_val = torch.max(layer).item()

        min_val = torch.min(layer).item()

        statistics.extend([
            mean_val,
            std_val,
            max_val,
            min_val
        ])

    return statistics

def scan_hidden_states(
    prompt,
    model,
    tokenizer,
    device
):

    print(
        "\n--- Hidden State Analysis ---"
    )

    print(f"Prompt: {prompt}")

    hidden_states = extract_hidden_states(
        prompt,
        model,
        tokenizer,
        device
    )

    transition_scores = (
        compute_layer_transitions(
            hidden_states
        )
    )

    activation_stats = (
        compute_activation_statistics(
            hidden_states
        )
    )

    transition_scores = np.array(
        transition_scores,
        dtype=np.float32
    )

    transition_scores = (
        transition_scores /
        (
            np.linalg.norm(
                transition_scores
            ) + 1e-8
        )
    )

    return {

        "transition_scores":
            transition_scores.tolist(),

        "activation_stats":
            activation_stats
    }

if __name__ == "__main__":

    from model_loader import (
        load_model_and_tokenizer
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model, tokenizer = (
        load_model_and_tokenizer()
    )

    prompt = (
        "How can malware evade detection?"
    )

    result = scan_hidden_states(
        prompt,
        model,
        tokenizer,
        device
    )

    print("\nTransition Scores:")

    print(
        result["transition_scores"]
    )

    print("\nActivation Stats:")

    print(
        result["activation_stats"][:10]
    )