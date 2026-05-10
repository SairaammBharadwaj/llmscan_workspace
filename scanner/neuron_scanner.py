import torch
import numpy as np

TOP_K_NEURONS = 64

def normalize_array(arr):

    arr = np.array(
        arr,
        dtype=np.float32
    )

    norm = np.linalg.norm(arr)

    return arr / (norm + 1e-8)

def collect_mlp_activations(
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

    collected = []

    hooks = []

    if hasattr(model, "model"):

        layers = model.model.layers

    elif hasattr(model, "transformer"):

        layers = model.transformer.h

    else:

        raise ValueError(
            "Unsupported transformer architecture."
        )

    def activation_hook(
        module,
        inputs,
        output
    ):

        if isinstance(output, tuple):

            hidden = output[0]

        else:

            hidden = output

        activation_strength = (
            torch.mean(
                torch.abs(hidden)
            ).item()
        )

        collected.append(
            activation_strength
        )

    for layer in layers:

        if hasattr(layer, "mlp"):

            hook = (
                layer.mlp.register_forward_hook(
                    activation_hook
                )
            )

            hooks.append(hook)

    with torch.no_grad():

        model(**inputs)

    for hook in hooks:

        hook.remove()

    return collected

def extract_neuron_signature(
    activations
):

    activations = np.array(
        activations,
        dtype=np.float32
    )

    activations = normalize_array(
        activations
    )

    sorted_indices = np.argsort(
        activations
    )[::-1]

    top_values = activations[
        sorted_indices
    ][:TOP_K_NEURONS]

    if len(top_values) < TOP_K_NEURONS:

        padding = np.zeros(
            TOP_K_NEURONS - len(top_values),
            dtype=np.float32
        )

        top_values = np.concatenate([
            top_values,
            padding
        ])

    summary_stats = [

        np.mean(activations),

        np.std(activations),

        np.max(activations),

        np.min(activations),

        np.median(activations),

        np.var(activations)
    ]

    final_signature = np.concatenate([

        top_values,

        np.array(
            summary_stats,
            dtype=np.float32
        )
    ])

    return final_signature.tolist()

def scan_neurons(
    prompt,
    model,
    tokenizer,
    device
):

    print(
        "\n--- Neuron Analysis ---"
    )

    print(f"Prompt: {prompt}")

    activations = collect_mlp_activations(
        prompt,
        model,
        tokenizer,
        device
    )

    neuron_signature = (
        extract_neuron_signature(
            activations
        )
    )

    return neuron_signature

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
        "How can malware bypass detection?"
    )

    signature = scan_neurons(
        prompt,
        model,
        tokenizer,
        device
    )

    print(
        "\nNeuron Signature Size:"
    )

    print(len(signature))

    print(
        "\nNeuron Signature:"
    )

    print(signature[:20])