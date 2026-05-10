import torch
import pandas as pd
import os
import json
import random
import numpy as np

from tqdm import tqdm

from scanner.token_scanner import scan_tokens
from scanner.layer_scanner import scan_layers
from scanner.hidden_state_scanner import (
    scan_hidden_states
)
from scanner.neuron_scanner import (
    scan_neurons
)
from scanner.feature_extractor import (
    extract_token_features
)
from scanner.model_loader import (
    load_model_and_tokenizer
)

device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

OUTPUT_FILE="data/advanced_causal_dataset_v4.csv"

MAX_DATASET_SIZE=2000

def clean_prompt(text):

    text=str(text).strip()

    text=text.replace("\n"," ")

    return text

def load_balanced_prompts(
    target_per_class=1
):

    base_path=(
        "data/processed_questions/"
        "combined_dataset"
    )

    all_safe=[]

    all_unsafe=[]

    for f_name in [
        "AutoDAN_orig.json",
        "GCG_orig.json",
        "PAP_orig.json"
    ]:

        with open(
            os.path.join(base_path,f_name),
            "r",
            encoding="utf-8"
        ) as f:

            data=json.load(f)

            all_unsafe.extend(
                data["adv_data"]
            )

            all_safe.extend(
                data["non_adv_data"]
            )

    social_files=[

        f for f in os.listdir(base_path)

        if (
            "SocialChem" in f
            and
            "Toxic" not in f
        )
    ]

    for f_name in social_files:

        with open(
            os.path.join(base_path,f_name),
            "r",
            encoding="utf-8"
        ) as f:

            data=json.load(f)

            all_unsafe.extend(
                data["toxic_data"]
            )

            all_safe.extend(
                data["non_toxic_data"]
            )

    bias_files=[

        f for f in os.listdir(base_path)

        if (
            "BBQ" in f
            and
            "Bias" not in f
        )
    ]

    for f_name in bias_files:

        with open(
            os.path.join(base_path,f_name),
            "r",
            encoding="utf-8"
        ) as f:

            data=json.load(f)

            all_unsafe.extend(
                data["stereotype_data"]
            )

            all_safe.extend(
                data["non_stereotype_data"]
            )

    safe_clean=list(set([
        clean_prompt(p)

        for p in all_safe

        if len(str(p)) > 15
    ]))

    unsafe_clean=list(set([
        clean_prompt(p)

        for p in all_unsafe

        if len(str(p)) > 15
    ]))

    safe_clean=random.sample(
        safe_clean,
        target_per_class
    )

    unsafe_clean=random.sample(
        unsafe_clean,
        target_per_class
    )

    return safe_clean,unsafe_clean

def build_feature_vector(
    token_scores,
    layer_scores,
    hidden_state_data,
    neuron_signature
):

    token_features=extract_token_features(
        token_scores
    )

    layer_scores=np.array(
        layer_scores,
        dtype=np.float32
    )

    layer_scores=(
        layer_scores /
        (
            np.linalg.norm(layer_scores)
            + 1e-8
        )
    )

    transition_scores=np.array(
        hidden_state_data[
            "transition_scores"
        ],
        dtype=np.float32
    )

    activation_stats=np.array(
        hidden_state_data[
            "activation_stats"
        ],
        dtype=np.float32
    )

    activation_summary=[

        np.mean(activation_stats),

        np.std(activation_stats),

        np.max(activation_stats),

        np.min(activation_stats),

        np.median(activation_stats),

        np.var(activation_stats),

        np.percentile(
            activation_stats,
            25
        ),

        np.percentile(
            activation_stats,
            75
        )
    ]

    neuron_signature=np.array(
        neuron_signature,
        dtype=np.float32
    )

    neuron_signature=(
        neuron_signature /
        (
            np.linalg.norm(
                neuron_signature
            ) + 1e-8
        )
    )

    final_vector=np.concatenate([

        np.array(
            token_features,
            dtype=np.float32
        ),

        layer_scores,

        transition_scores,

        np.array(
            activation_summary,
            dtype=np.float32
        ),

        neuron_signature
    ])

    return final_vector.tolist()

def run_massive_scan():

    print(
        "Loading Mistral-7B..."
    )

    llm,tokenizer=load_model_and_tokenizer()

    safe_list,unsafe_list=(
        load_balanced_prompts(
            target_per_class=1
        )
    )

    dataset=(
        [(p,0) for p in safe_list]
        +
        [(p,1) for p in unsafe_list]
    )

    random.shuffle(dataset)

    results=[]

    print(
        f"Starting Advanced Scan "
        f"on {len(dataset)} prompts..."
    )

    for idx,(prompt,label) in enumerate(
        tqdm(dataset)
    ):

        try:

            token_scores=scan_tokens(
                prompt,
                llm,
                tokenizer,
                device
            )

            layer_scores=scan_layers(
                prompt,
                llm,
                tokenizer,
                device
            )

            hidden_state_data=(
                scan_hidden_states(
                    prompt,
                    llm,
                    tokenizer,
                    device
                )
            )

            neuron_signature=scan_neurons(
                prompt,
                llm,
                tokenizer,
                device
            )

            feature_vector=build_feature_vector(
                token_scores,
                layer_scores,
                hidden_state_data,
                neuron_signature
            )

            feature_vector.append(label)

            results.append(feature_vector)

            if idx % 25 == 0 and idx > 0:

                df=pd.DataFrame(results)

                df.to_csv(
                    OUTPUT_FILE,
                    index=False
                )

                print(
                    f"Checkpoint Saved "
                    f"({idx} samples)"
                )

        except Exception as e:

            print(
                f"Skipping prompt "
                f"due to error: {e}"
            )

            continue

    df=pd.DataFrame(results)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nDataset Saved:"
    )

    print(OUTPUT_FILE)

    print(
        f"Total Samples: {len(results)}"
    )

if __name__=="__main__":

    run_massive_scan()