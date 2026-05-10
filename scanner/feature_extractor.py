import numpy as np
from scipy.stats import entropy

TOP_K_TOKENS = 32

def extract_token_features(token_distances):
    arr = np.array(token_distances, dtype=np.float32)

    if len(arr) == 0:
        return [0.0] * (TOP_K_TOKENS + 8)

    arr = np.abs(arr)

    arr = arr / (np.linalg.norm(arr) + 1e-8)

    sorted_arr = np.sort(arr)[::-1]

    top_k = sorted_arr[:TOP_K_TOKENS]

    if len(top_k) < TOP_K_TOKENS:
        padding = np.zeros(
            TOP_K_TOKENS - len(top_k),
            dtype=np.float32
        )

        top_k = np.concatenate([top_k, padding])

    mean_val = float(np.mean(arr))

    std_val = float(np.std(arr))

    max_val = float(np.max(arr))

    min_val = float(np.min(arr))

    range_val = float(np.ptp(arr))

    median_val = float(np.median(arr))

    variance_val = float(np.var(arr))

    prob_dist = arr / (np.sum(arr) + 1e-8)

    entropy_val = float(entropy(prob_dist))

    statistical_features = [
        mean_val,
        std_val,
        max_val,
        min_val,
        range_val,
        median_val,
        variance_val,
        entropy_val
    ]

    final_vector = np.concatenate([
        top_k,
        np.array(statistical_features, dtype=np.float32)
    ])

    return final_vector.tolist()

if __name__ == "__main__":

    sample_token_distances = [
        12.9375,
        1.8203,
        1.1406,
        1.7891,
        2.2812,
        1.0469,
        1.9453,
        1.2891
    ]

    features = extract_token_features(
        sample_token_distances
    )

    print(f"Feature Vector Size: {len(features)}")

    print(features)