import torch

def get_attention_scores(model, input_ids):

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True
        )

    if (
        outputs.attentions is None
        or len(outputs.attentions) == 0
    ):

        raise ValueError(
            "Model did not return attention tensors."
        )

    return outputs.attentions[0]

def scan_tokens(prompt, model, tokenizer, device):

    print(f"\n--- Starting Token Causal Scan ---")
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

    print("Extracting Baseline Attention...")

    baseline_attention = get_attention_scores(
        model,
        original_input_ids
    )

    token_causal_effects = []

    for i in range(num_tokens):

        intervened_ids = original_input_ids.clone()

        intervened_ids[0, i] = intervention_token_id

        intervened_attention = get_attention_scores(
            model,
            intervened_ids
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
            f"Intervened Token: "
            f"[{replaced_word.strip():<10}] "
            f"-> Causal Effect: {distance:.4f}"
        )

    return token_causal_effects