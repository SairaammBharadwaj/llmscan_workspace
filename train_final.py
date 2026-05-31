import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Keep training runs repeatable when PyTorch uses random initialization.
torch.manual_seed(42)

# Use the GPU when CUDA is available; otherwise run on the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store the training dataset path in one place so it is easy to change.
DATASET_PATH = "data/" "advanced_causal_dataset_v7.parquet"


class UltraCausalDetector(nn.Module):
    """Neural network that predicts whether a prompt looks unsafe."""

    def __init__(self, input_dim):
        # Initialize the base PyTorch module.
        super().__init__()
        # Build a feed-forward classifier that shrinks features down to one logit.
        self.network = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
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
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # Send the input features through every layer in the network.
        return self.network(x)


def compute_metrics(logits, labels):
    """Convert raw model outputs into common classification metrics."""
    # Turn logits into probabilities between 0 and 1.
    probs = torch.sigmoid(logits)
    # Convert probabilities into binary predictions using 0.5 as the cutoff.
    preds = (probs > 0.5).float()
    # Move labels to NumPy because sklearn metrics expect NumPy-like arrays.
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_prob = probs.cpu().detach().numpy()
    # Measure the percentage of correct predictions.
    accuracy = accuracy_score(y_true, y_pred)
    # Measure how many predicted unsafe samples were actually unsafe.
    precision = precision_score(y_true, y_pred, zero_division=0)
    # Measure how many real unsafe samples the model caught.
    recall = recall_score(y_true, y_pred, zero_division=0)
    # Combine precision and recall into one balanced score.
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Measure ranking quality across all probability thresholds.
    roc_auc = roc_auc_score(y_true, y_prob)
    # Return all metrics in a fixed order for printing later.
    return (accuracy, precision, recall, f1, roc_auc)


def load_dataset():
    """Load, clean, normalize, and split the detector training dataset."""
    # Stop early with a clear error if the dataset file is missing.
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found.")
    # Read the saved feature table from parquet.
    df = pd.read_parquet(DATASET_PATH)
    # Replace infinite values because the model cannot train on them.
    df = df.replace([np.inf, -np.inf], np.nan)
    # Remove rows that still contain missing values after cleanup.
    df = df.dropna()
    # Reset row numbers so the cleaned dataset has a tidy index.
    df = df.reset_index(drop=True)
    print("\nDataset Shape:")
    print(df.shape)
    # Use every column except prompt and label as model input features.
    X = df.iloc[:, :-2].values.astype(np.float32)
    # Use the last column as the target label.
    y = df.iloc[:, -1].values.astype(np.float32)
    # Save the feature count so the model input layer matches the dataset.
    input_dim = X.shape[1]
    # Convert feature and label arrays into PyTorch tensors.
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    # Replace any remaining NaN values with valid numbers.
    X = torch.nan_to_num(X)
    # Normalize each sample so scale differences do not dominate training.
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    # Split into training and validation sets while keeping class balance.
    return (
        train_test_split(X, y, test_size=0.2, random_state=42, stratify=y),
        input_dim,
    )


def train():
    """Train the detector and save the best model checkpoint."""
    # Load the split tensors and the feature dimension discovered from data.
    ((X_train, X_val, y_train, y_val), input_dim) = load_dataset()
    # Move training and validation tensors onto the selected device.
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)
    # Create the detector with the correct number of input features.
    model = UltraCausalDetector(input_dim).to(device)
    # AdamW updates model weights and applies mild weight decay regularization.
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    # Slowly lower the learning rate over the full training schedule.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    # Binary cross entropy with logits is the standard loss for one-logit binary classifiers.
    criterion = nn.BCEWithLogitsLoss()
    # Track the best validation loss so only the strongest checkpoint is saved.
    best_val_loss = float("inf")
    # Count epochs without improvement; currently kept for optional early stopping.
    patience_counter = 0
    # Number of stale epochs that would trigger early stopping if re-enabled.
    EARLY_STOPPING = 50
    # Maximum number of training passes over the dataset.
    EPOCHS = 300
    print(f"\nTraining on {device}")
    print(f"Input Dimension: {input_dim}")
    print(f"Train Samples: {len(X_train)}")
    print(f"Validation Samples: {len(X_val)}")
    # Train for the configured number of epochs.
    for epoch in range(EPOCHS):
        # Enable training behavior such as dropout.
        model.train()
        # Clear gradients from the previous epoch.
        optimizer.zero_grad()
        # Run the model on the training features.
        train_logits = model(X_train)
        # Compare predictions with true labels.
        train_loss = criterion(train_logits, y_train)
        # Compute gradients for every trainable parameter.
        train_loss.backward()
        # Prevent very large gradients from destabilizing training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Apply one optimizer update.
        optimizer.step()
        # Advance the learning-rate schedule.
        scheduler.step()
        # Switch to evaluation behavior before validating.
        model.eval()
        # Disable gradient tracking during validation to save memory.
        with torch.no_grad():
            # Run the model on the validation features.
            val_logits = model(X_val)
            # Measure validation loss.
            val_loss = criterion(val_logits, y_val)
            # Calculate metrics on the training predictions.
            (train_acc, train_prec, train_rec, train_f1, train_auc) = compute_metrics(
                train_logits, y_train
            )
            # Calculate metrics on the validation predictions.
            (val_acc, val_prec, val_rec, val_f1, val_auc) = compute_metrics(
                val_logits, y_val
            )
        # Save a checkpoint whenever validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs("detector", exist_ok=True)
            torch.save(model.state_dict(), "detector/misbehavior_detector.pth")
        else:
            # Count this as a non-improving epoch.
            patience_counter += 1
        # Print progress every five epochs to keep the console readable.
        if epoch % 5 == 0:
            print(f"\nEpoch [{epoch}/{EPOCHS}]")
            print(f"Train Loss: " f"{train_loss.item():.4f}")
            print(f"Val Loss: " f"{val_loss.item():.4f}")
            print(f"Train Acc: " f"{train_acc:.4f}")
            print(f"Val Acc: " f"{val_acc:.4f}")
            print(f"Val Precision: " f"{val_prec:.4f}")
            print(f"Val Recall: " f"{val_rec:.4f}")
            print(f"Val F1: " f"{val_f1:.4f}")
            print(f"Val ROC-AUC: " f"{val_auc:.4f}")
        # Early stopping is left here as an easy switch if shorter training is needed.
        # if patience_counter >= EARLY_STOPPING:
        #     print(
        #         "\nEarly stopping triggered."
        #     )
        #     break
    print("\nTraining Complete.")
    print("Best detector saved:")
    print("detector/misbehavior_detector.pth")


if __name__ == "__main__":
    # Run training only when this file is executed directly.
    train()
