import numpy as np
from scipy.stats import skew, kurtosis

def extract_token_features(token_distances):
    """
    Takes an array of token-level causal effects (which varies in length based on prompt length)
    and extracts a fixed 5-dimensional statistical feature vector.
    """
    # Convert to numpy array for optimized math operations
    arr = np.array(token_distances)
    
    # Safety check for empty arrays
    if len(arr) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # Calculate the 5 statistical features defined in the LLMScan paper
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    range_val = np.ptp(arr)  # ptp is "peak to peak" (Max - Min)
    
    # Skewness and Kurtosis require a minimum number of data points to be mathematically valid
    skew_val = float(skew(arr)) if len(arr) > 2 else 0.0
    kurt_val = float(kurtosis(arr)) if len(arr) > 3 else 0.0
    
    return [mean_val, std_val, range_val, skew_val, kurt_val]

# Quick local test block
if __name__ == "__main__":
    # Using the exact array you just generated from your token scanner!
    sample_token_distances = [12.9375, 1.8203, 1.1406, 1.7891, 2.2812, 1.0469, 1.9453, 1.2891]
    
    features = extract_token_features(sample_token_distances)
    
    print("Raw Token Distances:", sample_token_distances)
    print("\nExtracted 5D Feature Vector (Mean, Std, Range, Skew, Kurtosis):")
    print([round(f, 4) for f in features])