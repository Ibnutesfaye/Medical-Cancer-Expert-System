"""
Differential Privacy Training
===============================
Wraps the existing training pipeline with Opacus DP-SGD.
Adds mathematically calibrated Gaussian noise to gradients so
no individual patient's image can be identified from the model.

Usage:
    venv\\Scripts\\python.exe -m ai.dp_training

Key parameters:
    TARGET_EPSILON  — privacy budget (lower = more private, less accurate)
    TARGET_DELTA    — failure probability (1e-5 is standard)
    MAX_GRAD_NORM   — gradient clipping bound

Output:
    cancer_model_dp.pth   — differentially private model weights
    dp_metrics.json       — epsilon achieved, accuracy, privacy budget used
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

BASE          = Path(__file__).parent.parent
MODEL_PATH    = BASE / "cancer_model_dp.pth"
DP_METRICS    = BASE / "dp_metrics.json"

# ── DP Hyperparameters ────────────────────────────────────────────────────────
TARGET_EPSILON = 8.0    # ε — privacy budget. ε=8 → moderate privacy (~1-3% accuracy loss)
TARGET_DELTA   = 1e-5   # δ — probability of privacy failure
MAX_GRAD_NORM  = 1.0    # gradient clipping bound (C in DP-SGD)
EPOCHS         = 20     # fewer epochs needed with DP (noise handles regularization)
BATCH_SIZE     = 64     # larger batches improve DP utility
LR             = 1e-4


def train_with_dp():
    """
    Train ResNet18 with Differential Privacy using Opacus.
    Falls back to standard training if Opacus is not installed.
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        OPACUS_AVAILABLE = True
        print("  Opacus available — training with DP-SGD")
    except ImportError:
        OPACUS_AVAILABLE = False
        print("  Opacus not installed — run: pip install opacus==1.4.0")
        print("  Falling back to standard training (no privacy guarantees)")

    # Import shared training infrastructure
    import sys
    sys.path.insert(0, str(BASE))
    from train_model import (
        CancerDataset, build_file_index, TRAIN_TRANSFORM, VAL_TRANSFORM,
        LABELS_PATH, CSV_PATH, MAX_PER_CLASS
    )
    import pandas as pd
    from collections import defaultdict
    import torchvision.models as models

    print("=" * 60)
    print("Differential Privacy Training — DP-SGD")
    print(f"  ε (target) = {TARGET_EPSILON}")
    print(f"  δ (target) = {TARGET_DELTA}")
    print(f"  Grad norm  = {MAX_GRAD_NORM}")
    print("=" * 60)

    # Load and prepare dataset (same as train_model.py)
    df           = pd.read_csv(CSV_PATH)
    file_index   = build_file_index()

    valid_rows = []
    for _, row in df.iterrows():
        fname = str(row["image_name"])
        if fname in file_index:
            valid_rows.append({
                "image_name":  fname,
                "cancer_type": str(row["cancer_type"]),
                "label":       str(row["label"]),
            })

    class_buckets: dict[str, list] = defaultdict(list)
    for r in valid_rows:
        class_buckets[r["cancer_type"]].append(r)
    sampled: list[dict] = []
    for ct, rows in class_buckets.items():
        np.random.shuffle(rows)
        sampled.extend(rows[:MAX_PER_CLASS])
    valid_rows = sampled

    classes      = sorted(set(r["cancer_type"] for r in valid_rows))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes  = len(classes)

    np.random.shuffle(valid_rows)
    split      = int(len(valid_rows) * 0.9)
    train_rows = valid_rows[:split]
    val_rows   = valid_rows[split:]

    train_ds = CancerDataset(train_rows, file_index, class_to_idx, TRAIN_TRANSFORM)
    val_ds   = CancerDataset(val_rows,   file_index, class_to_idx, VAL_TRANSFORM)

    # DP requires batch_sampler=None and Poisson sampling
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    epsilon_achieved = None

    if OPACUS_AVAILABLE:
        # Make model DP-compatible (replaces BatchNorm with GroupNorm)
        model = ModuleValidator.fix(model)
        model = model.to(device)

        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=EPOCHS,
            target_epsilon=TARGET_EPSILON,
            target_delta=TARGET_DELTA,
            max_grad_norm=MAX_GRAD_NORM,
        )
        print(f"  Noise multiplier: {optimizer.noise_multiplier:.4f}")
    else:
        model = model.to(device)

    best_val_acc = 0.0
    results = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_correct, train_total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds       = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct / val_total if val_total > 0 else 0.0

        if OPACUS_AVAILABLE:
            epsilon_achieved = privacy_engine.get_epsilon(delta=TARGET_DELTA)
            print(f"Epoch {epoch}/{EPOCHS} — train={train_acc:.3f} val={val_acc:.3f} "
                  f"ε={epsilon_achieved:.2f} time={time.time()-t0:.1f}s")
        else:
            print(f"Epoch {epoch}/{EPOCHS} — train={train_acc:.3f} val={val_acc:.3f} "
                  f"time={time.time()-t0:.1f}s")

        results.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "train_acc":        train_acc,
                "num_classes":      num_classes,
                "class_to_idx":     class_to_idx,
                "dp_enabled":       OPACUS_AVAILABLE,
                "epsilon":          epsilon_achieved,
                "delta":            TARGET_DELTA,
            }, MODEL_PATH)
            print(f"  ✓ Best DP model saved (val_acc={val_acc:.3f})")

    # Save metrics
    metrics = {
        "dp_enabled":       OPACUS_AVAILABLE,
        "target_epsilon":   TARGET_EPSILON,
        "target_delta":     TARGET_DELTA,
        "epsilon_achieved": epsilon_achieved,
        "max_grad_norm":    MAX_GRAD_NORM,
        "best_val_acc":     best_val_acc,
        "epochs":           EPOCHS,
        "batch_size":       BATCH_SIZE,
        "training_curve":   results,
    }
    with open(DP_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDP Training complete.")
    print(f"  Best val_acc : {best_val_acc:.3f}")
    if epsilon_achieved:
        print(f"  ε achieved   : {epsilon_achieved:.2f} (target: {TARGET_EPSILON})")
    print(f"  Model saved  → {MODEL_PATH}")
    print(f"  Metrics saved → {DP_METRICS}")


if __name__ == "__main__":
    train_with_dp()
