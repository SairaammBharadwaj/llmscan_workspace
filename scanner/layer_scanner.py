import torch

def get_first_token_logits_batch(

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

                attention_mask=attention_mask
            )

            next_token_logits=(
                outputs.logits[:, -1, :]
            )

    return next_token_logits

def scan_layers_batch(

    prompts,
    model,
    tokenizer,
    device

):

    print(
        "\n--- Batch Layer Scan ---"
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

    baseline_logits=(
        get_first_token_logits_batch(

            model,

            input_ids,

            attention_mask
        )
    )

    if (
        hasattr(model,"model")
        and
        hasattr(model.model,"layers")
    ):

        layers=model.model.layers

    elif (
        hasattr(model,"transformer")
        and
        hasattr(model.transformer,"h")
    ):

        layers=model.transformer.h

    else:

        raise ValueError(
            "Unsupported architecture."
        )

    num_layers=len(layers)

    batch_size=input_ids.shape[0]

    all_results=[

        []

        for _ in range(batch_size)
    ]

    for layer_idx in range(num_layers):

        def skip_layer_hook(
            module,
            args,
            output
        ):

            if isinstance(output,tuple):

                new_output=(
                    args[0],
                ) + output[1:]

            else:

                new_output=args[0]

            return new_output

        target_layer=layers[layer_idx]

        hook_handle=(
            target_layer.register_forward_hook(
                skip_layer_hook
            )
        )

        try:

            intervened_logits=(
                get_first_token_logits_batch(

                    model,

                    input_ids,

                    attention_mask
                )
            )

            distances=torch.norm(

                baseline_logits
                -
                intervened_logits,

                p=2,

                dim=1
            )

            for batch_idx in range(
                batch_size
            ):

                all_results[
                    batch_idx
                ].append(

                    distances[
                        batch_idx
                    ].item()
                )

        finally:

            hook_handle.remove()

    return all_results