import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class MisbehaviorMLP(nn.Module):

    def __init__(self, input_size=37, hidden_size=64):

        super(MisbehaviorMLP, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 32)
        self.layer3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.relu(self.layer1(x))
        x = self.dropout(x)

        x = self.relu(self.layer2(x))
        x = self.dropout(x)

        x = self.sigmoid(self.layer3(x))

        return x

def run_training():

    csv_path = "data/gcg_training_maps_100.csv"

    if not os.path.exists(csv_path):

        print(f"❌ Error: Could not find {csv_path}")

        return

    print("Loading dataset...")

    df = pd.read_csv(csv_path)

    X_raw = df.iloc[:, :-1].values
    y_raw = df["label"].values

    print(f"Dataset Shape: {df.shape}")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_raw)

    X = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    y = torch.tensor(
        y_raw,
        dtype=torch.float32
    ).unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train = X_train.to(device)
    X_test = X_test.to(device)

    y_train = y_train.to(device)
    y_test = y_test.to(device)

    model = MisbehaviorMLP().to(device)

    for m in model.modules():

        if isinstance(m, nn.Linear):

            nn.init.xavier_uniform_(m.weight)

            nn.init.zeros_(m.bias)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print(f"\n🚀 Training on device: {device}")

    epochs = 500

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 50 == 0:

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {loss.item():.6f}"
            )

    print("\nEvaluating model...")

    model.eval()

    with torch.no_grad():

        predictions = model(X_test)

        predicted_labels = (
            predictions >= 0.5
        ).float()

    y_true = y_test.cpu().numpy()
    y_pred = predicted_labels.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print("\n========== RESULTS ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(cm)

    os.makedirs(
        "detector",
        exist_ok=True
    )

    model_path = (
        "detector/misbehavior_detector.pth"
    )

    scaler_path = (
        "detector/scaler.pkl"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print("\n✅ Model Saved:")
    print(model_path)

    print("\n✅ Scaler Saved:")
    print(scaler_path)

if __name__ == "__main__":

    run_training()