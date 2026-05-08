# scanner/layer_scanner.py

import torch

def get_first_token_logits(model, input_ids):

    with torch.no_grad():

        outputs = model(input_ids)

        next_token_logits = outputs.logits[0, -1, :]

    return next_token_logits

def scan_layers(prompt, model, tokenizer, device):

    print(f"\n--- Starting Layer Causal Scan ---")
    print(f"Prompt: '{prompt}'")

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

    input_ids = inputs["input_ids"]

    print("Extracting Baseline Logits...")

    baseline_logits = get_first_token_logits(
        model,
        input_ids
    )

    layer_causal_effects = []

    if hasattr(model, "model") and hasattr(model.model, "layers"):

        layers = model.model.layers

    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):

        layers = model.transformer.h

    else:

        raise ValueError(
            "Unsupported model architecture."
        )

    num_layers = len(layers)

    for layer_idx in range(num_layers):

        def skip_layer_hook(module, args, output):

            if isinstance(output, tuple):

                new_output = (args[0],) + output[1:]

            else:

                new_output = args[0]

            return new_output

        target_layer = layers[layer_idx]

        hook_handle = target_layer.register_forward_hook(
            skip_layer_hook
        )

        try:

            intervened_logits = get_first_token_logits(
                model,
                input_ids
            )

            distance = torch.norm(
                baseline_logits - intervened_logits,
                p=2
            ).item()

            layer_causal_effects.append(distance)

            print(
                f"Skipped Layer {layer_idx:02d} "
                f"-> Causal Effect: {distance:.4f}"
            )

        finally:

            hook_handle.remove()

    return layer_causal_effects

if __name__ == "__main__":

    from model_loader import load_model_and_tokenizer

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    test_model, test_tokenizer = load_model_and_tokenizer()

    test_prompt = "What is the capital of France?"

    distances = scan_layers(
        test_prompt,
        test_model,
        test_tokenizer,
        device
    )

    print("\nFinal Array of Layer Distances:")

    print([round(d, 4) for d in distances])