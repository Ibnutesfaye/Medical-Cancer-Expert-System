"""
Federated Learning Client
==========================
Represents one hospital node in the federated network.
Trains locally on hospital-specific data and sends only
model weights to the FL server — patient images never leave.

Usage:
    venv\\Scripts\\python.exe -m ai.federated.fl_client --hospital A --data-dir ./data_hospital_a
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, random_split

BASE = Path(__file__).parent.parent.parent


class CancerFederatedClient:
    """
    Flower NumPy client for federated cancer classification.
    Each hospital creates one instance with its local dataset.
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       torch.device,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.model.to(device)

    def get_parameters(self) -> list[np.ndarray]:
        """Return current model weights as flat NumPy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: list[np.ndarray]):
        """Load global model weights from server."""
        for p, w in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(w, dtype=p.dtype).to(self.device)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        """
        Receive global weights, train locally, return updated weights.
        Called by Flower server each round.
        """
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", 3))
        lr           = float(config.get("lr", 1e-4))

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(local_epochs):
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(imgs), labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple:
        """Evaluate global model on local validation data."""
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds    = self.model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += imgs.size(0)
        accuracy = correct / total if total > 0 else 0.0
        loss     = 1.0 - accuracy   # proxy loss
        return loss, total, {"accuracy": accuracy}


def start_fl_client(server_address: str = "localhost:8080", hospital: str = "A"):
    """Start a Flower client connecting to the FL server."""
    try:
        import flwr as fl
    except ImportError:
        print("Flower not installed. Run: pip install flwr==1.8.0")
        return

    import torchvision.models as models
    import sys
    sys.path.insert(0, str(BASE))
    from train_model import (
        CancerDataset, build_file_index,
        TRAIN_TRANSFORM, VAL_TRANSFORM, CSV_PATH, MAX_PER_CLASS
    )
    import pandas as pd
    from collections import defaultdict

    print(f"[Hospital {hospital}] Starting FL client → {server_address}")

    labels_path = BASE / "label_classes.json"
    with open(labels_path) as f:
        label_info = json.load(f)
    class_to_idx = label_info["class_to_idx"]
    num_classes  = len(class_to_idx)

    df         = pd.read_csv(CSV_PATH)
    file_index = build_file_index()

    valid_rows = []
    for _, row in df.iterrows():
        fname = str(row["image_name"])
        if fname in file_index:
            valid_rows.append({
                "image_name":  fname,
                "cancer_type": str(row["cancer_type"]),
                "label":       str(row["label"]),
            })

    # Simulate hospital partition: each hospital gets 1/3 of data
    buckets: dict[str, list] = defaultdict(list)
    for r in valid_rows:
        buckets[r["cancer_type"]].append(r)
    sampled = []
    hospital_idx = ord(hospital.upper()) - ord('A')   # A=0, B=1, C=2
    for rows in buckets.values():
        chunk_size = max(1, len(rows) // 3)
        start = hospital_idx * chunk_size
        sampled.extend(rows[start:start + chunk_size])

    np.random.shuffle(sampled)
    split      = int(len(sampled) * 0.9)
    train_rows = sampled[:split]
    val_rows   = sampled[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CancerDataset(train_rows, file_index, class_to_idx, TRAIN_TRANSFORM)
    val_ds   = CancerDataset(val_rows,   file_index, class_to_idx, VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    client = CancerFederatedClient(model, train_loader, val_loader, device)

    # Convert to Flower NumPyClient
    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config): return client.get_parameters()
        def fit(self, p, c):             return client.fit(p, c)
        def evaluate(self, p, c):        return client.evaluate(p, c)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=FlowerClient(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital",       default="A")
    parser.add_argument("--server-address", default="localhost:8080")
    args = parser.parse_args()
    start_fl_client(args.server_address, args.hospital)
