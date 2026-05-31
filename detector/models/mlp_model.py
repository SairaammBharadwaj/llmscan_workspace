import torch
import torch.nn as nn


class MisbehaviorDetector(nn.Module):
    def __init__(self, input_dim=37):
        """
        A lightweight Multi-Layer Perceptron (MLP) for classifying causal maps.
        input_dim default is 37 (5 token features + 32 layer features for Mistral-7B).
        """
        super(MisbehaviorDetector, self).__init__()
        # Define the neural network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Dropout helps prevent overfitting during training
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Squashes the final output to a probability between 0 and 1
        )

    def forward(self, x):
        """Passes the causal map through the network."""
        return self.network(x)


# Quick local test block
if __name__ == "__main__":
    # Initialize the model
    detector = MisbehaviorDetector(input_dim=37)
    # Let's simulate a "fake" causal map combining our previous test results
    # 5 token features + 32 layer features
    mock_token_features = [3.0312, 3.7655, 11.8906, 2.2169, 3.0148]
    mock_layer_features = [
        286.0,
        322.0,
        80.0,
        107.5,
        95.0,
        85.0,
        113.5,
        54.25,
        68.0,
        50.25,
        68.0,
        70.5,
        71.5,
        69.5,
        74.0,
        90.5,
        86.5,
        92.5,
        86.5,
        105.0,
        103.5,
        87.5,
        87.0,
        82.0,
        96.0,
        71.5,
        69.0,
        71.5,
        93.0,
        133.0,
        97.5,
        191.0,
    ]
    # Combine them into one list
    mock_causal_map = mock_token_features + mock_layer_features
    # Convert to a PyTorch tensor and add a batch dimension (1, 37)
    input_tensor = torch.tensor([mock_causal_map], dtype=torch.float32)
    # Run inference!
    with torch.no_grad():
        prediction = detector(input_tensor)
    probability = prediction.item()
    print("--- Testing the MLP Detector ---")
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Raw Output Probability: {probability:.4f}")
    if probability > 0.5:
        print("Classification: MISBEHAVIOR DETECTED 🚨")
    else:
        print("Classification: NORMAL BEHAVIOR ✅")
