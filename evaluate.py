import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

INPUT_DIM = 72

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

class AdvancedCausalDetector(nn.Module):

    def __init__(self):

        super().__init__()

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

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)

def load_model():

    model = AdvancedCausalDetector()

    state_dict = torch.load(
        "detector/misbehavior_detector.pth",
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model

def load_dataset():

    df = pd.read_csv(
        "data/advanced_causal_dataset_v2.csv"
    )

    X = df.iloc[:, :-1].values

    y = df.iloc[:, -1].values

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    X = torch.nn.functional.normalize(
        X,
        p=2,
        dim=1
    )

    y = torch.tensor(
        y,
        dtype=torch.float32
    )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def evaluate():

    model = load_model()

    _, X_test, _, y_test = load_dataset()

    X_test = X_test.to(device)

    with torch.no_grad():

        logits = model(X_test)

        probs = torch.sigmoid(
            logits
        ).cpu().numpy().flatten()

    preds = (probs > 0.5).astype(int)

    y_true = y_test.numpy()

    accuracy = accuracy_score(
        y_true,
        preds
    )

    precision = precision_score(
        y_true,
        preds
    )

    recall = recall_score(
        y_true,
        preds
    )

    f1 = f1_score(
        y_true,
        preds
    )

    roc_auc = roc_auc_score(
        y_true,
        probs
    )

    print("\n=== Evaluation Metrics ===")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\n=== Classification Report ===")

    print(
        classification_report(
            y_true,
            preds
        )
    )

    cm = confusion_matrix(
        y_true,
        preds
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix"
    )

    plt.colorbar()

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks([0, 1])

    plt.yticks([0, 1])

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.show()

    fpr, tpr, _ = roc_curve(
        y_true,
        probs
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.show()

if __name__ == "__main__":

    evaluate()