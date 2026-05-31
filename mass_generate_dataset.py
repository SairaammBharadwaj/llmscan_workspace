import torch
import pandas as pd
import os
import json
import random
import numpy as np

from tqdm import tqdm

from scanner.token_scanner import scan_tokens_batch
from scanner.layer_scanner import scan_layers_batch
from scanner.hidden_state_scanner import scan_hidden_states_batch
from scanner.neuron_scanner import scan_neurons_batch
from scanner.feature_extractor import extract_token_features
from scanner.model_loader import load_model_and_tokenizer
from scanner.semantic_scanner import extract_semantic_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_FILE = "data/advanced_causal_dataset_v7.parquet"

CHECKPOINT_EVERY = 10


def clean_prompt(text):
    text = str(text).strip()
    text = text.replace("\n", " ")
    return text


def normalize_group(arr):
    arr = np.array(arr, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        std = 1e-8
    arr = (arr - mean) / std
    arr = np.nan_to_num(arr)
    return arr


def load_balanced_prompts():
    base_path = "data/processed_questions/" "combined_dataset"
    all_safe = []
    all_unsafe = []
    for f_name in ["AutoDAN_orig.json", "GCG_orig.json", "PAP_orig.json"]:
        with open(os.path.join(base_path, f_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            all_unsafe.extend(data["adv_data"])
            all_safe.extend(data["non_adv_data"])
    social_files = [
        f for f in os.listdir(base_path) if ("SocialChem" in f and "Toxic" not in f)
    ]
    for f_name in social_files:
        with open(os.path.join(base_path, f_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            all_unsafe.extend(data["toxic_data"])
            all_safe.extend(data["non_toxic_data"])
    bias_files = [f for f in os.listdir(base_path) if ("BBQ" in f and "Bias" not in f)]
    for f_name in bias_files:
        with open(os.path.join(base_path, f_name), "r", encoding="utf-8") as f:
            data = json.load(f)
            all_unsafe.extend(data["stereotype_data"])
            all_safe.extend(data["non_stereotype_data"])
    dataset_files = [
        "data/hard_negatives.json",
        "data/cyber_hard_negatives.json",
        "data/programming_safe.json",
    ]
    for dataset_path in dataset_files:
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                extra_data = json.load(f)
                if "safe" in extra_data:
                    all_safe.extend(extra_data["safe"])
                if "unsafe" in extra_data:
                    all_unsafe.extend(extra_data["unsafe"])
    safe_clean = list(set([clean_prompt(p) for p in all_safe if len(str(p)) > 15]))
    unsafe_clean = list(set([clean_prompt(p) for p in all_unsafe if len(str(p)) > 15]))
    random.shuffle(safe_clean)
    random.shuffle(unsafe_clean)
    print(f"\nLoaded " f"{len(safe_clean)} SAFE prompts")
    print(f"Loaded " f"{len(unsafe_clean)} UNSAFE prompts")
    return safe_clean, unsafe_clean


def summarize_hidden_states(hidden_state_data):
    transition_scores = np.array(
        hidden_state_data["transition_scores"], dtype=np.float32
    )
    activation_stats = np.array(hidden_state_data["activation_stats"], dtype=np.float32)
    transition_scores = normalize_group(transition_scores)
    activation_summary = [
        np.mean(activation_stats),
        np.std(activation_stats),
        np.max(activation_stats),
        np.min(activation_stats),
        np.median(activation_stats),
        np.var(activation_stats),
        np.percentile(activation_stats, 25),
        np.percentile(activation_stats, 75),
    ]
    activation_summary = normalize_group(activation_summary)
    return (transition_scores, activation_summary)


def build_feature_vector(
    token_scores, layer_scores, hidden_state_data, neuron_signature, semantic_features
):
    token_features = extract_token_features(token_scores)
    token_features = normalize_group(token_features)
    layer_scores = normalize_group(layer_scores)
    (transition_scores, activation_summary) = summarize_hidden_states(hidden_state_data)
    neuron_signature = normalize_group(neuron_signature)
    semantic_features = normalize_group(semantic_features)
    final_vector = np.concatenate(
        [
            token_features,
            layer_scores,
            transition_scores,
            activation_summary,
            neuron_signature,
            semantic_features,
        ]
    )
    final_vector = np.nan_to_num(final_vector)
    final_vector = np.clip(final_vector, -10, 10)
    return final_vector.tolist()


def load_existing_dataset():
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_parquet(OUTPUT_FILE)
            existing_data = existing_df.values.tolist()
            print(f"Loaded existing dataset " f"with {len(existing_data)} samples")
            return existing_data
        except Exception as e:
            print(f"Could not load existing " f"dataset: {e}")
            return []
    return []


def run_massive_scan():
    print("\nLoading Mistral-7B...")
    llm, tokenizer = load_model_and_tokenizer()
    safe_list, unsafe_list = load_balanced_prompts()
    dataset = [(p, 0) for p in safe_list] + [(p, 1) for p in unsafe_list]
    random.shuffle(dataset)
    total_prompts = len(dataset)
    print(f"\nTOTAL PROMPTS:" f" {total_prompts}")
    results = load_existing_dataset()
    existing_prompts = set()
    try:
        existing_df = pd.read_parquet(OUTPUT_FILE)
        existing_prompts = set(existing_df.iloc[:, -2].astype(str).tolist())
    except Exception as e:
        print(f"Could not rebuild " f"existing prompts: {e}")
    completed = len(existing_prompts)
    remaining = total_prompts - completed
    print(f"\nAlready Completed:" f" {completed}")
    print(f"Remaining:" f" {remaining}")
    progress_bar = tqdm(total=remaining, desc="Scanning Prompts", unit="prompt")
    processed_count = 0
    for idx, (prompt, label) in enumerate(dataset):
        try:
            if prompt in existing_prompts:
                continue
            token_scores = scan_tokens_batch([prompt], llm, tokenizer, device)[0]
            layer_scores = scan_layers_batch([prompt], llm, tokenizer, device)[0]
            hidden_state_data = scan_hidden_states_batch(
                [prompt], llm, tokenizer, device
            )[0]
            neuron_signature = scan_neurons_batch([prompt], llm, tokenizer, device)[0]
            semantic_features = extract_semantic_features(prompt)
            feature_vector = build_feature_vector(
                token_scores,
                layer_scores,
                hidden_state_data,
                neuron_signature,
                semantic_features,
            )
            feature_vector.append(prompt)
            feature_vector.append(label)
            results.append(feature_vector)
            processed_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"processed": processed_count, "remaining": remaining - processed_count}
            )
            if processed_count % CHECKPOINT_EVERY == 0:
                df = pd.DataFrame(results)
                df.to_parquet(OUTPUT_FILE, index=False)
                print(f"\nCheckpoint Saved " f"({processed_count})")
        except Exception as e:
            print(f"\nSkipping prompt " f"due to error: {e}")
            continue
    progress_bar.close()
    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_FILE, index=False)
    print("\nFINAL DATASET SAVED")
    print(OUTPUT_FILE)
    print(f"Total Samples: " f"{len(results)}")


if __name__ == "__main__":
    run_massive_scan()
