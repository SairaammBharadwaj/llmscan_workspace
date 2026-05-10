import numpy as np

def normalize_group(features):

    arr=np.array(
        features,
        dtype=np.float32
    )

    mean=np.mean(arr)

    std=np.std(arr)

    if std < 1e-8:

        std=1e-8

    normalized=(
        arr - mean
    ) / std

    return normalized.tolist()

def normalize_feature_groups(

    token_features,
    layer_features,
    hidden_features,
    neuron_features

):

    token_features=normalize_group(
        token_features
    )

    layer_features=normalize_group(
        layer_features
    )

    hidden_features=normalize_group(
        hidden_features
    )

    neuron_features=normalize_group(
        neuron_features
    )

    combined=(
        token_features +
        layer_features +
        hidden_features +
        neuron_features
    )

    return combined