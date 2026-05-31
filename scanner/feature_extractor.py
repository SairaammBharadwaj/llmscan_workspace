import numpy as np
from scipy.stats import entropy

TOP_K_TOKENS = 32


def extract_token_features(token_distances):
    """Convert variable-length token distances into a fixed-size feature vector."""
    # Convert scanner output into a float array.
    arr = np.array(token_distances, dtype=np.float32)
    # Return an all-zero vector when no token distances are available.
    if len(arr) == 0:
        return [0.0] * (TOP_K_TOKENS + 8)
    # Use absolute distance values because magnitude matters more than sign here.
    arr = np.abs(arr)
    # Normalize distances so prompt length does not dominate the scale.
    arr = arr / (np.linalg.norm(arr) + 1e-8)
    # Sort largest distances first so top-k features are consistent.
    sorted_arr = np.sort(arr)[::-1]
    # Keep only the strongest token distances.
    top_k = sorted_arr[:TOP_K_TOKENS]
    # Pad short prompts so every feature vector has the same length.
    if len(top_k) < TOP_K_TOKENS:
        padding = np.zeros(TOP_K_TOKENS - len(top_k), dtype=np.float32)
        top_k = np.concatenate([top_k, padding])
    # Calculate summary statistics that describe the full token-distance shape.
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    max_val = float(np.max(arr))
    min_val = float(np.min(arr))
    range_val = float(np.ptp(arr))
    median_val = float(np.median(arr))
    variance_val = float(np.var(arr))
    # Convert distances into a probability-like distribution for entropy.
    prob_dist = arr / (np.sum(arr) + 1e-8)
    # Entropy measures how spread out the token-distance signal is.
    entropy_val = float(entropy(prob_dist))
    # Keep all summary statistics in a stable order.
    statistical_features = [
        mean_val,
        std_val,
        max_val,
        min_val,
        range_val,
        median_val,
        variance_val,
        entropy_val,
    ]
    # Join top-k distances and summary statistics into one vector.
    final_vector = np.concatenate(
        [top_k, np.array(statistical_features, dtype=np.float32)]
    )
    # Return a plain Python list for easy storage in pandas/parquet.
    return final_vector.tolist()


if __name__ == "__main__":
    # Example values let this module be tested directly from the command line.
    sample_token_distances = [
        12.9375,
        1.8203,
        1.1406,
        1.7891,
        2.2812,
        1.0469,
        1.9453,
        1.2891,
    ]
    # Build features for the example token distances.
    features = extract_token_features(sample_token_distances)
    # Print the output size and values for inspection.
    print(f"Feature Vector Size: {len(features)}")
    print(features)
