import torch

def get_attention_scores_batch(

    model,
    input_ids,
    attention_mask

):

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs=model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                output_attentions=True,

                return_dict=True
            )

    if (
        outputs.attentions is None
        or
        len(outputs.attentions)==0
    ):

        raise ValueError(
            "No attention tensors returned."
        )

    return outputs.attentions[0]

def scan_tokens_batch(

    prompts,
    model,
    tokenizer,
    device

):

    print(
        "\n--- Batch Token Scan ---"
    )

    inputs=tokenizer(

        prompts,

        return_tensors="pt",

        padding=True,

        truncation=True,

        max_length=128

    )

    inputs={

        k:v.to(device)

        for k,v in inputs.items()
    }

    input_ids=inputs["input_ids"]

    attention_mask=inputs[
        "attention_mask"
    ]

    baseline_attention=(
        get_attention_scores_batch(

            model,

            input_ids,

            attention_mask
        )
    )

    batch_size=input_ids.shape[0]

    seq_len=input_ids.shape[1]

    intervention_token_id=(
        tokenizer.convert_tokens_to_ids("-")
    )

    batch_results=[]

    for batch_idx in range(batch_size):

        token_effects=[]

        for token_idx in range(seq_len):

            if (
                attention_mask[
                    batch_idx,
                    token_idx
                ].item()==0
            ):

                continue

            intervened_ids=input_ids.clone()

            intervened_ids[
                batch_idx,
                token_idx
            ]=intervention_token_id

            intervened_attention=(
                get_attention_scores_batch(

                    model,

                    intervened_ids,

                    attention_mask
                )
            )

            distance=torch.norm(

                baseline_attention[
                    batch_idx
                ]
                -
                intervened_attention[
                    batch_idx
                ],

                p=2

            ).item()

            token_effects.append(
                distance
            )

        batch_results.append(
            token_effects
        )

    return batch_results