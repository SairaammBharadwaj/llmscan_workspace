# generate_dataset.py

import torch
import pandas as pd
import os
import time
import json

from model_loader import load_model_and_tokenizer
from token_scanner import scan_tokens
from layer_scanner import scan_layers
from feature_extractor import extract_token_features

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def create_training_data(
    safe_prompts,
    unsafe_prompts,
    output_filename="data/training_causal_maps.csv"
):

    print("Loading Model...")

    model, tokenizer = load_model_and_tokenizer()

    print(f"Using device: {device}")

    all_data = []

    print("\n--- Scanning SAFE Prompts ---")

    for idx, prompt in enumerate(safe_prompts):

        try:

            print(
                f"Safe [{idx+1}/{len(safe_prompts)}]: "
                f"{prompt[:50]}..."
            )

            token_dist = scan_tokens(
                prompt,
                model,
                tokenizer,
                device
            )

            layer_dist = scan_layers(
                prompt,
                model,
                tokenizer,
                device
            )

            token_feat = extract_token_features(
                token_dist
            )

            causal_map = (
                token_feat +
                layer_dist +
                [0]
            )

            all_data.append(causal_map)

        except Exception as e:

            print(f"SAFE prompt failed: {e}")

    print("\n--- Scanning UNSAFE Prompts ---")

    for idx, prompt in enumerate(unsafe_prompts):

        try:

            print(
                f"Unsafe [{idx+1}/{len(unsafe_prompts)}]: "
                f"{prompt[:50]}..."
            )

            token_dist = scan_tokens(
                prompt,
                model,
                tokenizer,
                device
            )

            layer_dist = scan_layers(
                prompt,
                model,
                tokenizer,
                device
            )

            token_feat = extract_token_features(
                token_dist
            )

            causal_map = (
                token_feat +
                layer_dist +
                [1]
            )

            all_data.append(causal_map)

        except Exception as e:

            print(f"UNSAFE prompt failed: {e}")

    print("\nSaving dataset...")

    columns = (
        [f"token_feat_{i}" for i in range(5)] +
        [f"layer_dist_{i}" for i in range(32)] +
        ["label"]
    )

    df = pd.DataFrame(
        all_data,
        columns=columns
    )

    os.makedirs(
        os.path.dirname(output_filename),
        exist_ok=True
    )

    df.to_csv(
        output_filename,
        index=False
    )

    print(
        f"Dataset successfully saved "
        f"to {output_filename}!"
    )

if __name__ == "__main__":

    unsafe_path = (
        "data1/data/data/processed_questions/"
        "combined_dataset/AutoDAN_orig.json"
    )

    safe_path = (
        "data1/data/data/processed_questions/"
        "commonsense_QA_v2_dev.json"
    )

    print("Loading datasets...")

    with open(
        unsafe_path,
        "r",
        encoding="utf-8"
    ) as f:

        unsafe_raw = json.load(f)

        if "adv_data" in unsafe_raw:
            unsafe_prompts = unsafe_raw["adv_data"]
        else:
            unsafe_prompts = []

    with open(
        safe_path,
        "r",
        encoding="utf-8"
    ) as f:

        safe_raw = json.load(f)

        if "question" in safe_raw:
            safe_prompts = list(
                safe_raw["question"].values()
            )
        else:
            safe_prompts = []

    clean_safe = []
    clean_unsafe = []

    for text in unsafe_prompts:

        if (
            isinstance(text, str)
            and len(text.strip()) > 20
        ):
            clean_unsafe.append(
                text.strip()
            )

    for text in safe_prompts:

        if (
            isinstance(text, str)
            and len(text.strip()) > 20
        ):
            clean_safe.append(
                text.strip()
            )

    clean_safe = list(set(clean_safe))
    clean_unsafe = list(set(clean_unsafe))

    print(
        f"Cleaned and sorted into "
        f"{len(clean_safe)} Safe Prompts "
        f"and {len(clean_unsafe)} Unsafe Prompts."
    )

    if (
        len(clean_safe) >= 50
        and len(clean_unsafe) >= 50
    ):

        safe_test_batch = clean_safe[:50]
        unsafe_test_batch = clean_unsafe[:50]

        print(
            f"\n🚀 Firing up the "
            f"LLMScan Pipeline for "
            f"{len(safe_test_batch) + len(unsafe_test_batch)} prompts..."
        )

        start_time = time.time()

        output_csv = (
            "data1/gcg_training_maps_100.csv"
        )

        create_training_data(
            safe_test_batch,
            unsafe_test_batch,
            output_filename=output_csv
        )

        print(
            f"\nFinished in "
            f"{time.time() - start_time:.2f} seconds."
        )

    else:

        print(
            "Error: Still not finding "
            "the prompts. Check the file paths!"
        )