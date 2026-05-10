import torch

def get_logits(
    model,
    input_ids
):

    with torch.no_grad():

        outputs=model(
            input_ids=input_ids
        )

    logits=outputs.logits[
        0,
        -1,
        :
    ]

    return logits

def scan_layers(
    prompt,
    model,
    tokenizer,
    device
):

    print(
        "\n--- Starting Advanced Layer Scan ---"
    )

    print(f"Prompt: '{prompt}'")

    inputs=tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs={
        k:v.to(device)
        for k,v in inputs.items()
    }

    input_ids=inputs["input_ids"]

    baseline_logits=get_logits(
        model,
        input_ids
    )

    if hasattr(model,"model"):

        layers=model.model.layers

    elif hasattr(model,"transformer"):

        layers=model.transformer.h

    else:

        raise ValueError(
            "Unsupported model architecture."
        )

    layer_effects=[]

    for idx,layer in enumerate(layers):

        def identity_hook(
            module,
            inputs,
            output
        ):

            return inputs[0]

        hook=layer.register_forward_hook(
            identity_hook
        )

        try:

            intervened_logits=get_logits(
                model,
                input_ids
            )

            distance=torch.norm(
                baseline_logits -
                intervened_logits,
                p=2
            ).item()

            layer_effects.append(
                distance
            )

            print(
                f"Layer {idx:02d} "
                f"-> CE: {distance:.6f}"
            )

        except Exception as e:

            print(
                f"Layer {idx} failed: {e}"
            )

            layer_effects.append(0.0)

        finally:

            hook.remove()

    layer_effects=torch.tensor(
        layer_effects,
        dtype=torch.float32
    )

    layer_effects=(
        layer_effects /
        (
            torch.norm(
                layer_effects,
                p=2
            ) + 1e-8
        )
    )

    return layer_effects.tolist()