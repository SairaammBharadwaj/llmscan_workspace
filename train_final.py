import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

torch.manual_seed(42)

device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

INPUT_DIM=182

class UltraCausalDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.network=nn.Sequential(

            nn.Linear(INPUT_DIM,768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.45),

            nn.Linear(768,512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.40),

            nn.Linear(512,256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(128,64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64,32),
            nn.GELU(),

            nn.Linear(32,1)
        )

    def forward(self,x):

        return self.network(x)

def compute_metrics(
    logits,
    labels
):

    probs=torch.sigmoid(logits)

    preds=(probs > 0.5).float()

    y_true=labels.cpu().numpy()

    y_pred=preds.cpu().numpy()

    y_prob=probs.cpu().detach().numpy()

    accuracy=accuracy_score(
        y_true,
        y_pred
    )

    precision=precision_score(
        y_true,
        y_pred
    )

    recall=recall_score(
        y_true,
        y_pred
    )

    f1=f1_score(
        y_true,
        y_pred
    )

    roc_auc=roc_auc_score(
        y_true,
        y_prob
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    )

def load_dataset():

    csv_path=(
        "data/"
        "advanced_causal_dataset_v4.csv"
    )

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"{csv_path} not found."
        )

    df=pd.read_csv(csv_path)

    X=df.iloc[:, :-1].values

    y=df.iloc[:, -1].values

    X=torch.tensor(
        X,
        dtype=torch.float32
    )

    y=torch.tensor(
        y,
        dtype=torch.float32
    ).unsqueeze(1)

    X=torch.nn.functional.normalize(
        X,
        p=2,
        dim=1
    )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def train():

    X_train,X_val,y_train,y_val=(
        load_dataset()
    )

    X_train=X_train.to(device)
    X_val=X_val.to(device)

    y_train=y_train.to(device)
    y_val=y_val.to(device)

    model=UltraCausalDetector().to(device)

    optimizer=optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=1e-4
    )

    scheduler=(
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=300
        )
    )

    criterion=nn.BCEWithLogitsLoss()

    best_val_loss=float("inf")

    patience_counter=0

    EARLY_STOPPING=50

    EPOCHS=300

    print(
        f"\nTraining on {device}"
    )

    print(
        f"Train Samples: {len(X_train)}"
    )

    print(
        f"Validation Samples: {len(X_val)}"
    )

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        train_logits=model(X_train)

        train_loss=criterion(
            train_logits,
            y_train
        )

        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        scheduler.step()

        model.eval()

        with torch.no_grad():

            val_logits=model(X_val)

            val_loss=criterion(
                val_logits,
                y_val
            )

            (
                train_acc,
                train_prec,
                train_rec,
                train_f1,
                train_auc
            )=compute_metrics(
                train_logits,
                y_train
            )

            (
                val_acc,
                val_prec,
                val_rec,
                val_f1,
                val_auc
            )=compute_metrics(
                val_logits,
                y_val
            )

        if val_loss < best_val_loss:

            best_val_loss=val_loss

            patience_counter=0

            os.makedirs(
                "detector",
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                "detector/misbehavior_detector.pth"
            )

        else:

            patience_counter += 1

        if epoch % 5 == 0:

            print(
                f"\nEpoch [{epoch}/{EPOCHS}]"
            )

            print(
                f"Train Loss: "
                f"{train_loss.item():.4f}"
            )

            print(
                f"Val Loss: "
                f"{val_loss.item():.4f}"
            )

            print(
                f"Train Acc: "
                f"{train_acc:.4f}"
            )

            print(
                f"Val Acc: "
                f"{val_acc:.4f}"
            )

            print(
                f"Val Precision: "
                f"{val_prec:.4f}"
            )

            print(
                f"Val Recall: "
                f"{val_rec:.4f}"
            )

            print(
                f"Val F1: "
                f"{val_f1:.4f}"
            )

            print(
                f"Val ROC-AUC: "
                f"{val_auc:.4f}"
            )

        if patience_counter >= EARLY_STOPPING:

            print(
                "\nEarly stopping triggered."
            )

            break

    print(
        "\nTraining Complete."
    )

    print(
        "Best detector saved:"
    )

    print(
        "detector/misbehavior_detector.pth"
    )

if __name__=="__main__":

    train()