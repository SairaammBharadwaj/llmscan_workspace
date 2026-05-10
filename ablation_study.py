import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

DATASET_PATH=(
    "data/advanced_causal_dataset_v4.csv"
)

INPUT_DIM=182

FEATURE_GROUPS={

    "token_features":(0,40),

    "layer_features":(40,72),

    "hidden_transitions":(72,104),

    "activation_summary":(104,112),

    "neuron_signature":(112,182)
}

class AblationDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.network=nn.Sequential(

            nn.Linear(INPUT_DIM,512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512,256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.GELU(),

            nn.Linear(64,1)
        )

    def forward(self,x):

        return self.network(x)

def load_dataset():

    df=pd.read_csv(DATASET_PATH)

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

def apply_ablation(
    X,
    feature_name
):

    X_modified=X.clone()

    start,end=FEATURE_GROUPS[
        feature_name
    ]

    X_modified[:, start:end]=0.0

    return X_modified

def train_and_evaluate(
    X_train,
    X_val,
    y_train,
    y_val
):

    model=AblationDetector().to(device)

    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=0.0005
    )

    criterion=nn.BCEWithLogitsLoss()

    EPOCHS=50

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        logits=model(X_train)

        loss=criterion(
            logits,
            y_train
        )

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        val_logits=model(X_val)

        probs=torch.sigmoid(
            val_logits
        )

        preds=(probs > 0.5).float()

    y_true=y_val.cpu().numpy()

    y_pred=preds.cpu().numpy()

    acc=accuracy_score(
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

    return {

        "accuracy":acc,

        "precision":precision,

        "recall":recall,

        "f1":f1
    }

def run_ablation():

    X_train,X_val,y_train,y_val=(
        load_dataset()
    )

    X_train=X_train.to(device)
    X_val=X_val.to(device)

    y_train=y_train.to(device)
    y_val=y_val.to(device)

    print(
        "\n=== BASELINE MODEL ==="
    )

    baseline_results=(
        train_and_evaluate(
            X_train,
            X_val,
            y_train,
            y_val
        )
    )

    print(baseline_results)

    results={

        "baseline":
            baseline_results
    }

    for feature_group in FEATURE_GROUPS:

        print(
            f"\n=== ABLATING: "
            f"{feature_group} ==="
        )

        X_train_mod=apply_ablation(
            X_train,
            feature_group
        )

        X_val_mod=apply_ablation(
            X_val,
            feature_group
        )

        ablation_results=(
            train_and_evaluate(
                X_train_mod,
                X_val_mod,
                y_train,
                y_val
            )
        )

        results[
            feature_group
        ]=ablation_results

        print(ablation_results)

    print(
        "\n=== FINAL ABLATION SUMMARY ==="
    )

    for key,val in results.items():

        print(f"\n{key}")

        for metric,mval in val.items():

            print(
                f"{metric}: "
                f"{mval:.4f}"
            )

if __name__=="__main__":

    run_ablation()