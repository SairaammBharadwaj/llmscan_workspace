import torch
import numpy as np

def scan_neurons_batch(

    prompts,
    model,
    tokenizer,
    device,

    top_k=32

):

    print(
        "\n--- Batch Neuron Scan ---"
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

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs=model(

                **inputs,

                output_hidden_states=True,

                return_dict=True
            )

    hidden_states=outputs.hidden_states

    final_hidden=hidden_states[-1]

    batch_size=final_hidden.shape[0]

    batch_results=[]

    for batch_idx in range(batch_size):

        hidden_vector=final_hidden[
            batch_idx
        ]

        neuron_strengths=(

            hidden_vector
            .abs()
            .mean(dim=0)
        )

        top_values,_=torch.topk(

            neuron_strengths,

            k=min(
                top_k,
                neuron_strengths.shape[0]
            )
        )

        neuron_signature=(
            top_values
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

        batch_results.append(
            neuron_signature
        )

    return batch_results