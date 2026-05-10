import torch
import numpy as np

MAX_TOKENS = 32

def get_attention_scores(
    model,
    input_ids
):

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True
        )

    attentions = outputs.attentions

    if attentions is None:
        raise ValueError(
            "No attention tensors returned."
        )

    stacked = torch.stack(attentions)

    aggregated = stacked.mean(dim=0)

    return aggregated

def normalize_attention(attn):

    attn = attn.float()

    norm = torch.norm(attn, p=2)

    return attn / (norm + 1e-8)

def scan_tokens(
    prompt,
    model,
    tokenizer,
    device
):

    print(f"\n--- Starting Advanced Token Causal Scan ---")

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

    original_input_ids = inputs["input_ids"]

    num_tokens = original_input_ids.shape[1]

    intervention_token_id = tokenizer.convert_tokens_to_ids("-")

    baseline_attention = get_attention_scores(
        model,
        original_input_ids
    )

    baseline_attention = normalize_attention(
        baseline_attention
    )

    token_causal_effects = []

    for i in range(num_tokens):

        intervened_ids = original_input_ids.clone()

        intervened_ids[0, i] = intervention_token_id

        intervened_attention = get_attention_scores(
            model,
            intervened_ids
        )

        intervened_attention = normalize_attention(
            intervened_attention
        )

        distance = torch.norm(
            baseline_attention -
            intervened_attention,
            p=2
        ).item()

        token_causal_effects.append(distance)

        replaced_word = tokenizer.decode(
            original_input_ids[0, i]
        )

        print(
            f"Token [{replaced_word.strip():<12}] "
            f"-> CE: {distance:.6f}"
        )

    token_causal_effects = np.array(
        token_causal_effects,
        dtype=np.float32
    )

    if len(token_causal_effects) > MAX_TOKENS:

        strongest_indices = np.argsort(
            token_causal_effects
        )[::-1][:MAX_TOKENS]

        token_causal_effects = token_causal_effects[
            strongest_indices
        ]

    return token_causal_effects.tolist()

if __name__ == "__main__":

    from model_loader import (
        load_model_and_tokenizer
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, tokenizer = load_model_and_tokenizer()

    prompt = "How to bypass WiFi security?"

    scores = scan_tokens(
        prompt,
        model,
        tokenizer,
        device
    )

    print("\nFinal Token Scores:")

    print(scores)