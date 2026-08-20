"""
Train a cancer detection model from dataset.csv + image folders.

Usage:
    venv/Scripts/python.exe train_model.py

Output:
    cancer_model.pth            — trained ResNet18 weights
    label_classes.json          — class index → cancer_type mapping
    metrics.json                — full evaluation metrics (multi-class + binary)
    confusion_matrix.png        — 14×14 multi-class confusion matrix heatmap
    confusion_matrix_binary.png — binary cancer vs non-cancer heatmap

Accuracy improvements applied:
    1. More epochs (30) with early stopping (patience=7)
    2. Higher per-class cap (500) → more training data
    3. Stronger augmentation (vertical flip, sharpness, perspective)
    4. Class-weighted loss → penalises rare classes more
    5. Cosine annealing LR scheduler → better convergence
    6. Differential learning rates → backbone lr/10, head lr full
    7. Label smoothing (0.1) → reduces overconfidence
    8. Mixed precision training (AMP) → faster on GPU
    9. Gradient clipping → stable training
   10. ResNet50 backbone option (better capacity than ResNet18)
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms

# scikit-learn evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Config  ← TUNE THESE TO INCREASE ACCURACY
# ---------------------------------------------------------------------------

BASE        = Path(__file__).parent
CSV_PATH    = BASE / "dataset.csv"
MODEL_PATH  = BASE / "cancer_model.pth"
LABELS_PATH = BASE / "label_classes.json"
METRICS_PATH   = BASE / "metrics.json"
CM_IMAGE_PATH  = BASE / "confusion_matrix.png"
CM_BINARY_PATH = BASE / "confusion_matrix_binary.png"

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 30       # was 25 → more epochs = more learning
LR            = 1e-4     # head learning rate
BACKBONE_LR   = 1e-5     # backbone gets 10× smaller lr (fine-tuning)
IMG_SIZE      = 224
NUM_WORKERS   = 0        # Windows-safe (set to 4 on Linux/Mac)
MAX_PER_CLASS = 500      # was 300 → more data per class = better generalisation
LABEL_SMOOTH  = 0.1      # label smoothing → reduces overconfidence
GRAD_CLIP     = 1.0      # gradient clipping → stable training
EARLY_STOP_PATIENCE = 7  # stop if val_acc doesn't improve for 7 epochs

# ── Backbone choice ──────────────────────────────────────────────────────────
# "resnet18"  → faster, less memory  (~79% baseline)
# "resnet50"  → better accuracy, needs more GPU memory (~85%+ expected)
BACKBONE = "resnet50"

# Cancer class names (positive class = 1 in binary evaluation)
CANCER_CLASSES = {
    "glioma_tumor",
    "meningioma_tumor",
    "pituitary_tumor",
    "malignant_lung_cancer",
    "melanoma",
    "basal_cell_carcinoma",
    "actinic_keratosis",
}

# Non-cancer class names (negative class = 0 in binary evaluation)
NON_CANCER_CLASSES = {
    "no_tumor",
    "benign_lung",
    "normal_lung",
    "melanocytic_nevi",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
}

# Image dirs to search for files
IMAGE_DIRS = [
    BASE / "brain cancer csv" / "Testing",
    BASE / "brain cancer csv" / "Training",
    BASE / "lung cancer csv" / "The IQ-OTHNCCD lung cancer dataset"
         / "The IQ-OTHNCCD lung cancer dataset",
    BASE / "skin cancer csv" / "HAM10000_images_part_1",
    BASE / "skin cancer csv" / "HAM10000_images_part_2",
]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # slightly larger then crop
    transforms.RandomCrop(IMG_SIZE),                    # random crop → position invariance
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),                    # NEW: vertical flip
    transforms.RandomRotation(20),                      # was 15 → wider rotation
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,                   # was 0.2
        saturation=0.2, hue=0.05,                       # NEW: saturation + hue
    ),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # NEW: sharpness
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),    # NEW: perspective
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # NEW: random erasing
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Build filename → path index
# ---------------------------------------------------------------------------

def build_file_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in IMAGE_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[p.name] = p
    print(f"  File index: {len(index)} images found on disk")
    return index

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CancerDataset(Dataset):
    def __init__(self, rows: list[dict], file_index: dict[str, Path],
                 class_to_idx: dict[str, int], transform):
        self.rows        = rows
        self.file_index  = file_index
        self.class_to_idx = class_to_idx
        self.transform   = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row      = self.rows[idx]
        img_path = self.file_index[row["image_name"]]
        with Image.open(img_path) as img:
            img = img.convert("RGB")   # normalise all modes (RGBA, P, L …)
        img   = self.transform(img)
        label = self.class_to_idx[row["cancer_type"]]
        return img, label

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def to_binary(class_names: list[str], indices: list[int]) -> list[int]:
    """
    Convert multi-class label indices to binary:
        cancer     → 1
        non-cancer → 0
    """
    return [1 if class_names[i] in CANCER_CLASSES else 0 for i in indices]


def save_confusion_matrix_heatmap(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    save_path: Path,
    figsize: tuple = (14, 11),
) -> None:
    """Render a labelled confusion matrix heatmap and save as PNG."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix image saved → {save_path}")


def evaluate_and_save(
    all_true: list[int],
    all_pred: list[int],
    idx_to_class: dict[int, str],
    train_acc: float,
    val_acc: float,
    train_loss: float,
    best_epoch: int,
) -> dict:
    """
    Compute full multi-class and binary evaluation metrics,
    print a detailed report, save metrics.json, and save heatmap PNGs.

    Returns the metrics dict.
    """
    sep = "=" * 60
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # ── Multi-class metrics ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("MULTI-CLASS EVALUATION  (14 classes)")
    print(sep)

    mc_accuracy  = accuracy_score(all_true, all_pred)
    mc_precision = precision_score(all_true, all_pred, average="weighted",
                                   zero_division=0)
    mc_recall    = recall_score(all_true, all_pred, average="weighted",
                                zero_division=0)
    mc_f1        = f1_score(all_true, all_pred, average="weighted",
                            zero_division=0)

    print(f"  Accuracy  : {mc_accuracy  * 100:.2f}%")
    print(f"  Precision : {mc_precision * 100:.2f}%  (weighted avg)")
    print(f"  Recall    : {mc_recall    * 100:.2f}%  (weighted avg)")
    print(f"  F1 Score  : {mc_f1        * 100:.2f}%  (weighted avg)")

    # Full per-class report
    print(f"\n{'─' * 60}")
    print("Per-class Classification Report:")
    print(classification_report(
        all_true, all_pred,
        target_names=class_names,
        zero_division=0,
    ))

    # 14×14 confusion matrix
    mc_cm = confusion_matrix(all_true, all_pred)
    print("14×14 Confusion Matrix:")
    print(mc_cm)

    # Save 14×14 heatmap
    save_confusion_matrix_heatmap(
        mc_cm,
        labels=class_names,
        title="Multi-Class Confusion Matrix (14 Classes)",
        save_path=CM_IMAGE_PATH,
        figsize=(16, 13),
    )

    # ── Binary metrics (cancer vs non-cancer) ────────────────────────────────
    print(f"\n{sep}")
    print("BINARY EVALUATION  (Cancer=1  vs  Non-Cancer=0)")
    print(sep)

    bin_true = to_binary(class_names, all_true)
    bin_pred = to_binary(class_names, all_pred)

    bin_accuracy  = accuracy_score(bin_true, bin_pred)
    bin_precision = precision_score(bin_true, bin_pred, zero_division=0)
    bin_recall    = recall_score(bin_true, bin_pred, zero_division=0)
    bin_f1        = f1_score(bin_true, bin_pred, zero_division=0)

    bin_cm = confusion_matrix(bin_true, bin_pred)

    # Extract TP / FP / FN / TN safely
    if bin_cm.shape == (2, 2):
        tn, fp, fn, tp = bin_cm.ravel()
    else:
        # Edge case: only one class present in val set
        tp = fp = fn = tn = 0

    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0   # False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Positive Rate

    print(f"  Accuracy          : {bin_accuracy  * 100:.2f}%")
    print(f"  Precision         : {bin_precision * 100:.2f}%")
    print(f"  Recall (Sensitivity): {bin_recall  * 100:.2f}%")
    print(f"  F1 Score          : {bin_f1        * 100:.2f}%")
    print(f"  False Negative Rate (FNR): {fnr    * 100:.2f}%  ← missed cancers")
    print(f"  False Positive Rate (FPR): {fpr    * 100:.2f}%  ← false alarms")
    print(f"\n  Confusion Matrix (Binary):")
    print(f"  {'':20s}  Pred: Non-Cancer  Pred: Cancer")
    print(f"  {'True: Non-Cancer':20s}  TN={tn:<10}  FP={fp}")
    print(f"  {'True: Cancer':20s}  FN={fn:<10}  TP={tp}")

    print(f"\n  Binary Classification Report:")
    print(classification_report(
        bin_true, bin_pred,
        target_names=["Non-Cancer", "Cancer"],
        zero_division=0,
    ))

    # Save binary heatmap
    save_confusion_matrix_heatmap(
        bin_cm,
        labels=["Non-Cancer", "Cancer"],
        title="Binary Confusion Matrix (Cancer vs Non-Cancer)",
        save_path=CM_BINARY_PATH,
        figsize=(6, 5),
    )

    # ── Medical safety assessment ────────────────────────────────────────────
    print(f"\n{sep}")
    print("MEDICAL SAFETY ASSESSMENT")
    print(sep)
    if bin_recall >= 0.95:
        verdict = "EXCELLENT — Safe for research screening (recall ≥ 95%)"
    elif bin_recall >= 0.88:
        verdict = "GOOD — Acceptable for research use (recall ≥ 88%)"
    elif bin_recall >= 0.80:
        verdict = "MODERATE — Use with caution, physician review required"
    else:
        verdict = "DANGEROUS — Too many missed cancers, do NOT use clinically"

    print(f"  Recall  : {bin_recall * 100:.2f}%")
    print(f"  FN Rate : {fnr * 100:.2f}%  ({fn} cancers missed out of {tp+fn})")
    print(f"  Verdict : {verdict}")

    # ── Build metrics dict ───────────────────────────────────────────────────
    metrics = {
        "training": {
            "best_epoch":   best_epoch,
            "train_accuracy": round(train_acc, 4),
            "val_accuracy":   round(val_acc,   4),
            "train_loss":     round(train_loss, 4),
        },
        "multiclass": {
            "accuracy":  round(mc_accuracy,  4),
            "precision": round(mc_precision, 4),
            "recall":    round(mc_recall,    4),
            "f1_score":  round(mc_f1,        4),
            "num_classes": len(class_names),
            "class_names": class_names,
            "confusion_matrix": mc_cm.tolist(),
        },
        "binary": {
            "accuracy":           round(bin_accuracy,  4),
            "precision":          round(bin_precision, 4),
            "recall":             round(bin_recall,    4),
            "f1_score":           round(bin_f1,        4),
            "false_negative_rate": round(fnr,          4),
            "false_positive_rate": round(fpr,          4),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "confusion_matrix": bin_cm.tolist(),
        },
        "safety": {
            "verdict": verdict,
            "fnr_pct": round(fnr * 100, 2),
            "missed_cancers": int(fn),
            "total_cancer_samples": int(tp + fn),
        },
    }

    # Save metrics.json
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {METRICS_PATH}")

    return metrics

# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train():
    print("=" * 60)
    print("Cancer Detection Model Training")
    print("=" * 60)

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"CSV loaded: {len(df)} rows")
    print(df["label"].value_counts().to_string())

    # Build file index
    file_index = build_file_index()

    # Filter to rows where image file exists on disk
    valid_rows = []
    for _, row in df.iterrows():
        fname = str(row["image_name"])
        if fname in file_index:
            valid_rows.append({
                "image_name":  fname,
                "cancer_type": str(row["cancer_type"]),
                "label":       str(row["label"]),
            })

    print(f"Valid rows (image found): {len(valid_rows)} / {len(df)}")

    if len(valid_rows) < 10:
        print("ERROR: Not enough images found. Check dataset folders.")
        return

    # Stratified cap: max MAX_PER_CLASS per cancer_type → fast + balanced
    class_buckets: dict[str, list] = defaultdict(list)
    for r in valid_rows:
        class_buckets[r["cancer_type"]].append(r)

    sampled: list[dict] = []
    for ct, rows in class_buckets.items():
        np.random.shuffle(rows)
        sampled.extend(rows[:MAX_PER_CLASS])
    valid_rows = sampled
    print(f"Stratified sample: {len(valid_rows)} rows "
          f"(max {MAX_PER_CLASS}/class × {len(class_buckets)} classes)")

    # Build class index
    classes       = sorted(set(r["cancer_type"] for r in valid_rows))
    class_to_idx  = {c: i for i, c in enumerate(classes)}
    idx_to_class  = {i: c for c, i in class_to_idx.items()}
    num_classes   = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # Save label mapping
    label_info = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "cancer_labels": {
            c: (df[df["cancer_type"] == c]["label"].iloc[0]
                if len(df[df["cancer_type"] == c]) > 0 else "non-cancer")
            for c in classes
        },
    }
    with open(LABELS_PATH, "w") as f:
        json.dump(label_info, f, indent=2)
    print(f"Label map saved → {LABELS_PATH}")

    # Train/val split (90/10)
    np.random.shuffle(valid_rows)
    split      = int(len(valid_rows) * 0.9)
    train_rows = valid_rows[:split]
    val_rows   = valid_rows[split:]
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    train_ds = CancerDataset(train_rows, file_index, class_to_idx, TRAIN_TRANSFORM)
    val_ds   = CancerDataset(val_rows,   file_index, class_to_idx, VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # ── Model — backbone selection ───────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Backbone: {BACKBONE}")

    if BACKBONE == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # ── Class-weighted loss ──────────────────────────────────────────────────
    # Rare classes (benign_lung=120, dermatofibroma=115) get higher weight
    class_counts = np.array([
        sum(1 for r in train_rows if r["cancer_type"] == idx_to_class[i])
        for i in range(num_classes)
    ], dtype=np.float32)
    class_counts   = np.maximum(class_counts, 1)
    class_weights  = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    class_weights  = class_weights / class_weights.sum() * num_classes

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTH,
    )
    print(f"Loss    : CrossEntropyLoss (weighted + label_smoothing={LABEL_SMOOTH})")

    # ── Differential learning rates ──────────────────────────────────────────
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params     = list(model.fc.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": LR},
    ], weight_decay=1e-4)

    # Cosine annealing → smooth lr decay to near-zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )

    # Mixed precision scaler (GPU only — no-op on CPU)
    scaler = GradScaler(enabled=device.type == "cuda")

    print(f"Optimizer: AdamW  backbone_lr={BACKBONE_LR}  head_lr={LR}")
    print(f"Scheduler: CosineAnnealingLR  T_max={EPOCHS}")

    best_val_acc    = 0.0
    best_epoch      = 1
    best_train_acc  = 0.0
    best_train_loss = 0.0
    no_improve      = 0          # early stopping counter

    # These will hold the predictions from the BEST epoch's validation pass
    best_val_true: list[int] = []
    best_val_pred: list[int] = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Training loop (with AMP + gradient clipping) ─────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(enabled=device.type == "cuda"):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # Gradient clipping → prevents exploding gradients
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={train_loss/train_total:.4f} "
                      f"acc={train_correct/train_total:.3f}")

        # ── Validation loop — collect ALL predictions for metrics ────────────
        model.eval()
        val_correct = 0
        val_total   = 0
        epoch_true: list[int] = []   # ground-truth label indices
        epoch_pred: list[int] = []   # predicted label indices

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds   = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

                # Accumulate for sklearn metrics
                epoch_true.extend(labels.cpu().tolist())
                epoch_pred.extend(preds.cpu().tolist())

        val_acc    = val_correct / val_total if val_total > 0 else 0.0
        train_acc  = train_correct / train_total if train_total > 0 else 0.0
        t_loss     = train_loss / train_total if train_total > 0 else 0.0
        elapsed    = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS} — "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
              f"time={elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc    = val_acc
            best_epoch      = epoch
            best_train_acc  = train_acc
            best_train_loss = t_loss
            no_improve      = 0          # reset early stopping counter

            # Keep the predictions from this best epoch for final evaluation
            best_val_true = epoch_true
            best_val_pred = epoch_pred

            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          val_acc,
                "train_acc":        train_acc,
                "train_loss":       t_loss,
                "num_classes":      num_classes,
                "class_to_idx":     class_to_idx,
            }, MODEL_PATH)
            print(f"  ✓ Best model saved (val_acc={val_acc:.3f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP_PATIENCE})")

        scheduler.step()

        # ── Early stopping ───────────────────────────────────────────────────
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\nTraining complete. Best val_acc={best_val_acc:.3f} "
          f"(epoch {best_epoch})")
    print(f"Model saved → {MODEL_PATH}")

    # ── Final evaluation on best-epoch validation predictions ────────────────
    print("\nRunning final evaluation on best checkpoint predictions …")
    evaluate_and_save(
        all_true    = best_val_true,
        all_pred    = best_val_pred,
        idx_to_class = idx_to_class,
        train_acc   = best_train_acc,
        val_acc     = best_val_acc,
        train_loss  = best_train_loss,
        best_epoch  = best_epoch,
    )


if __name__ == "__main__":
    train()
