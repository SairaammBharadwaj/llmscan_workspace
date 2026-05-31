import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# Number of feature columns expected by the saved detector.
INPUT_DIM = 72

# Use CUDA if it exists so evaluation can run faster.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvancedCausalDetector(nn.Module):
    """Detector architecture used by the saved evaluation checkpoint."""

    def __init__(self):
        # Initialize the PyTorch module base class.
        super().__init__()
        # Define the feed-forward network that maps features to one risk logit.
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Pass the input tensor through the detector network.
        return self.network(x)


def load_model():
    """Load the trained detector checkpoint from disk."""
    # Create a model object with the same architecture used during training.
    model = AdvancedCausalDetector()
    # Load saved weights and map them to the active CPU/GPU device.
    state_dict = torch.load("detector/misbehavior_detector.pth", map_location=device)
    # Copy the saved weights into the model.
    model.load_state_dict(state_dict)
    # Move the model to the selected device.
    model.to(device)
    # Disable training-only behavior such as dropout.
    model.eval()
    # Return the ready-to-use model.
    return model


def load_dataset():
    """Load the evaluation dataset and split it into train/test parts."""
    # Read the CSV file that contains feature columns and the label column.
    df = pd.read_csv("data/advanced_causal_dataset_v2.csv")
    # Treat all columns except the last one as input features.
    X = df.iloc[:, :-1].values
    # Treat the last column as the true label.
    y = df.iloc[:, -1].values
    # Convert features into float tensors for PyTorch.
    X = torch.tensor(X, dtype=torch.float32)
    # Normalize each row so features have a stable scale.
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    # Convert labels into float tensors.
    y = torch.tensor(y, dtype=torch.float32)
    # Return a stratified split so class balance is preserved.
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate():
    """Evaluate the detector and display metrics plus plots."""
    # Load the trained model checkpoint.
    model = load_model()
    # Keep only the test split for evaluation.
    _, X_test, _, y_test = load_dataset()
    # Move test features to the same device as the model.
    X_test = X_test.to(device)
    # Run inference without building a gradient graph.
    with torch.no_grad():
        # Get raw model outputs for every test example.
        logits = model(X_test)
        # Convert logits into probabilities and move them back to NumPy.
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    # Convert probabilities into binary predictions.
    preds = (probs > 0.5).astype(int)
    # Convert labels to NumPy for sklearn metrics.
    y_true = y_test.numpy()
    # Calculate standard classifier metrics.
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    roc_auc = roc_auc_score(y_true, probs)
    # Print the main metric values.
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    # Print the per-class precision, recall, and F1 scores.
    print("\n=== Classification Report ===")
    print(classification_report(y_true, preds))
    # Build the confusion matrix so mistakes are easy to inspect.
    cm = confusion_matrix(y_true, preds)
    # Draw the confusion matrix.
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    # Put the count inside each confusion-matrix cell.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    # Show the confusion matrix window.
    plt.show()
    # Calculate ROC curve points for plotting.
    fpr, tpr, _ = roc_curve(y_true, probs)
    # Draw the ROC curve.
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Run evaluation only when this file is executed directly.
    evaluate()
