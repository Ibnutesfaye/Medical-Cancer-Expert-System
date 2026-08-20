"""
Standalone Model Evaluation Script
===================================
Loads the trained cancer_model.pth and runs full evaluation
WITHOUT retraining. Generates all confusion matrices and metrics.

Usage:
    venv\\Scripts\\python.exe evaluate_model.py

Outputs:
    metrics.json                  — all metrics in JSON
    confusion_matrix.png          — 14×14 multi-class heatmap
    confusion_matrix_binary.png   — 2×2 cancer vs non-cancer heatmap
    confusion_matrix_brain.png    — brain cancer only heatmap
    confusion_matrix_lung.png     — lung cancer only heatmap
    confusion_matrix_skin.png     — skin cancer only heatmap
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE        = Path(__file__).parent
MODEL_PATH  = BASE / "cancer_model.pth"
LABELS_PATH = BASE / "label_classes.json"
CSV_PATH    = BASE / "dataset.csv"

METRICS_PATH    = BASE / "metrics.json"
CM_ALL_PATH     = BASE / "confusion_matrix.png"
CM_BINARY_PATH  = BASE / "confusion_matrix_binary.png"
CM_BRAIN_PATH   = BASE / "confusion_matrix_brain.png"
CM_LUNG_PATH    = BASE / "confusion_matrix_lung.png"
CM_SKIN_PATH    = BASE / "confusion_matrix_skin.png"

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

CANCER_CLASSES = {
    "glioma_tumor", "meningioma_tumor", "pituitary_tumor",
    "malignant_lung_cancer",
    "melanoma", "basal_cell_carcinoma", "actinic_keratosis",
}

NON_CANCER_CLASSES = {
    "no_tumor",
    "benign_lung", "normal_lung",
    "melanocytic_nevi", "benign_keratosis", "dermatofibroma", "vascular_lesion",
}

# Domain groupings for per-domain confusion matrices
BRAIN_CLASSES = {
    "glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"
}
LUNG_CLASSES  = {
    "malignant_lung_cancer", "benign_lung", "normal_lung"
}
SKIN_CLASSES  = {
    "melanoma", "basal_cell_carcinoma", "actinic_keratosis",
    "melanocytic_nevi", "benign_keratosis", "dermatofibroma", "vascular_lesion",
}

# Image dirs
IMAGE_DIRS = [
    BASE / "brain cancer csv" / "Testing",
    BASE / "brain cancer csv" / "Training",
    BASE / "lung cancer csv" / "The IQ-OTHNCCD lung cancer dataset"
         / "The IQ-OTHNCCD lung cancer dataset",
    BASE / "skin cancer csv" / "HAM10000_images_part_1",
    BASE / "skin cancer csv" / "HAM10000_images_part_2",
]

# Inference transform — must match training (no Grayscale)
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CancerDataset(Dataset):
    def __init__(self, rows, file_index, class_to_idx, transform):
        self.rows         = rows
        self.file_index   = file_index
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(self.file_index[row["image_name"]]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.class_to_idx[row["cancer_type"]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_file_index():
    index = {}
    for root in IMAGE_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[p.name] = p
    print(f"  File index: {len(index)} images found on disk")
    return index


def to_binary(class_names, indices):
    """Convert multi-class indices → 1 (cancer) / 0 (non-cancer)."""
    return [1 if class_names[i] in CANCER_CLASSES else 0 for i in indices]


def save_heatmap(cm, labels, title, save_path, figsize=(14, 11)):
    """Save a labelled confusion matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate():
    # ── Load model checkpoint ────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print("ERROR: cancer_model.pth not found. Run train_model.py first.")
        return

    print_section("Loading Model")
    checkpoint   = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    num_classes  = checkpoint["num_classes"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(num_classes)]

    saved_val_acc   = checkpoint.get("val_acc",   0.0)
    saved_train_acc = checkpoint.get("train_acc", 0.0)
    saved_loss      = checkpoint.get("train_loss", 0.0)
    saved_epoch     = checkpoint.get("epoch", "?")

    print(f"  Classes      : {num_classes}")
    print(f"  Best epoch   : {saved_epoch}")
    print(f"  Train acc    : {saved_train_acc * 100:.2f}%")
    print(f"  Val acc      : {saved_val_acc   * 100:.2f}%")
    print(f"  Train loss   : {saved_loss:.4f}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"  Device       : {device}")

    # ── Build evaluation dataset ─────────────────────────────────────────────
    print_section("Building Evaluation Dataset")
    df         = pd.read_csv(CSV_PATH)
    file_index = build_file_index()

    valid_rows = []
    for _, row in df.iterrows():
        fname = str(row["image_name"])
        if fname in file_index and str(row["cancer_type"]) in class_to_idx:
            valid_rows.append({
                "image_name":  fname,
                "cancer_type": str(row["cancer_type"]),
                "label":       str(row["label"]),
            })

    print(f"  Valid rows: {len(valid_rows)} / {len(df)}")

    # Use 10% as evaluation set (same split ratio as training)
    np.random.seed(42)          # fixed seed → reproducible split
    np.random.shuffle(valid_rows)
    split    = int(len(valid_rows) * 0.9)
    eval_rows = valid_rows[split:]
    print(f"  Evaluation set: {len(eval_rows)} images (10% split, seed=42)")

    eval_ds     = CancerDataset(eval_rows, file_index, class_to_idx, EVAL_TRANSFORM)
    eval_loader = DataLoader(eval_ds, batch_size=32, shuffle=False, num_workers=0)

    # ── Run inference ────────────────────────────────────────────────────────
    print_section("Running Inference")
    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []

    t0 = time.time()
    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            confs  = probs.max(dim=1).values

            all_true.extend(labels.tolist())
            all_pred.extend(preds.cpu().tolist())
            all_conf.extend(confs.cpu().tolist())

    print(f"  Inference done: {len(all_true)} images in {time.time()-t0:.1f}s")
    print(f"  Mean confidence: {np.mean(all_conf)*100:.1f}%")

    # ── 1. FULL 14-CLASS EVALUATION ──────────────────────────────────────────
    print_section("1 — FULL 14-CLASS EVALUATION")

    mc_acc  = accuracy_score(all_true, all_pred)
    mc_prec = precision_score(all_true, all_pred, average="weighted", zero_division=0)
    mc_rec  = recall_score(all_true, all_pred, average="weighted", zero_division=0)
    mc_f1   = f1_score(all_true, all_pred, average="weighted", zero_division=0)

    print(f"  Accuracy  : {mc_acc  * 100:.2f}%")
    print(f"  Precision : {mc_prec * 100:.2f}%  (weighted)")
    print(f"  Recall    : {mc_rec  * 100:.2f}%  (weighted)")
    print(f"  F1 Score  : {mc_f1   * 100:.2f}%  (weighted)")

    print(f"\n  Per-Class Report:")
    print(classification_report(
        all_true, all_pred, target_names=class_names, zero_division=0
    ))

    mc_cm = confusion_matrix(all_true, all_pred)
    print("  14×14 Confusion Matrix:")
    print(mc_cm)

    save_heatmap(
        mc_cm, class_names,
        "14-Class Confusion Matrix — All Cancer Types",
        CM_ALL_PATH, figsize=(16, 13),
    )

    # ── 2. BINARY EVALUATION (cancer vs non-cancer) ──────────────────────────
    print_section("2 — BINARY EVALUATION  (Cancer=1 vs Non-Cancer=0)")

    bin_true = to_binary(class_names, all_true)
    bin_pred = to_binary(class_names, all_pred)

    bin_acc  = accuracy_score(bin_true, bin_pred)
    bin_prec = precision_score(bin_true, bin_pred, zero_division=0)
    bin_rec  = recall_score(bin_true, bin_pred, zero_division=0)
    bin_f1   = f1_score(bin_true, bin_pred, zero_division=0)
    bin_cm   = confusion_matrix(bin_true, bin_pred)

    tn, fp, fn, tp = bin_cm.ravel() if bin_cm.shape == (2, 2) else (0, 0, 0, 0)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"  Accuracy          : {bin_acc  * 100:.2f}%")
    print(f"  Precision         : {bin_prec * 100:.2f}%")
    print(f"  Recall            : {bin_rec  * 100:.2f}%")
    print(f"  F1 Score          : {bin_f1   * 100:.2f}%")
    print(f"  False Negative Rate (FNR): {fnr * 100:.2f}%  ← missed cancers")
    print(f"  False Positive Rate (FPR): {fpr * 100:.2f}%  ← false alarms")
    print(f"\n  Binary Confusion Matrix:")
    print(f"  {'':22s}  Pred: Non-Cancer  Pred: Cancer")
    print(f"  {'True: Non-Cancer':22s}  TN={tn:<12}  FP={fp}")
    print(f"  {'True: Cancer':22s}  FN={fn:<12}  TP={tp}")
    print(f"\n  Binary Classification Report:")
    print(classification_report(
        bin_true, bin_pred,
        target_names=["Non-Cancer", "Cancer"], zero_division=0,
    ))

    save_heatmap(
        bin_cm, ["Non-Cancer", "Cancer"],
        "Binary Confusion Matrix — Cancer vs Non-Cancer",
        CM_BINARY_PATH, figsize=(6, 5),
    )

    # ── 3. BRAIN CANCER ONLY ─────────────────────────────────────────────────
    print_section("3 — BRAIN CANCER ONLY")
    _domain_eval(
        all_true, all_pred, class_names,
        domain_classes=BRAIN_CLASSES,
        domain_name="Brain Cancer",
        save_path=CM_BRAIN_PATH,
    )

    # ── 4. LUNG CANCER ONLY ──────────────────────────────────────────────────
    print_section("4 — LUNG CANCER ONLY")
    _domain_eval(
        all_true, all_pred, class_names,
        domain_classes=LUNG_CLASSES,
        domain_name="Lung Cancer",
        save_path=CM_LUNG_PATH,
    )

    # ── 5. SKIN CANCER ONLY ──────────────────────────────────────────────────
    print_section("5 — SKIN CANCER ONLY")
    _domain_eval(
        all_true, all_pred, class_names,
        domain_classes=SKIN_CLASSES,
        domain_name="Skin Cancer",
        save_path=CM_SKIN_PATH,
    )

    # ── Medical safety verdict ───────────────────────────────────────────────
    print_section("MEDICAL SAFETY ASSESSMENT")
    if bin_rec >= 0.95:
        verdict = "EXCELLENT — Safe for research screening (recall ≥ 95%)"
    elif bin_rec >= 0.88:
        verdict = "GOOD — Acceptable for research use (recall ≥ 88%)"
    elif bin_rec >= 0.80:
        verdict = "MODERATE — Use with caution, physician review required"
    else:
        verdict = "DANGEROUS — Too many missed cancers, do NOT use clinically"

    print(f"  Recall    : {bin_rec * 100:.2f}%")
    print(f"  FN Rate   : {fnr * 100:.2f}%  ({fn} cancers missed out of {tp+fn})")
    print(f"  Verdict   : {verdict}")

    # ── Save metrics.json ────────────────────────────────────────────────────
    metrics = {
        "model": {
            "best_epoch":   saved_epoch,
            "train_accuracy": round(saved_train_acc, 4),
            "val_accuracy":   round(saved_val_acc,   4),
            "train_loss":     round(saved_loss,       4),
        },
        "multiclass": {
            "accuracy":  round(mc_acc,  4),
            "precision": round(mc_prec, 4),
            "recall":    round(mc_rec,  4),
            "f1_score":  round(mc_f1,   4),
            "num_classes": num_classes,
            "class_names": class_names,
            "confusion_matrix": mc_cm.tolist(),
        },
        "binary": {
            "accuracy":            round(bin_acc,  4),
            "precision":           round(bin_prec, 4),
            "recall":              round(bin_rec,  4),
            "f1_score":            round(bin_f1,   4),
            "false_negative_rate": round(fnr,      4),
            "false_positive_rate": round(fpr,      4),
            "TP": int(tp), "FP": int(fp),
            "FN": int(fn), "TN": int(tn),
            "confusion_matrix": bin_cm.tolist(),
        },
        "safety": {
            "verdict":              verdict,
            "fnr_pct":              round(fnr * 100, 2),
            "missed_cancers":       int(fn),
            "total_cancer_samples": int(tp + fn),
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Final summary ────────────────────────────────────────────────────────
    print_section("SUMMARY — ALL OUTPUT FILES")
    outputs = [
        (METRICS_PATH,   "All metrics in JSON"),
        (CM_ALL_PATH,    "14-class confusion matrix heatmap"),
        (CM_BINARY_PATH, "Binary confusion matrix heatmap"),
        (CM_BRAIN_PATH,  "Brain cancer confusion matrix"),
        (CM_LUNG_PATH,   "Lung cancer confusion matrix"),
        (CM_SKIN_PATH,   "Skin cancer confusion matrix"),
    ]
    for path, desc in outputs:
        status = "✓" if path.exists() else "✗"
        print(f"  {status}  {path.name:<40}  {desc}")

    print(f"\n  Open the PNG files to view all confusion matrices.")
    print(f"  They are saved in:  {BASE}")


# ---------------------------------------------------------------------------
# Domain-specific evaluation helper
# ---------------------------------------------------------------------------

def _domain_eval(
    all_true, all_pred, class_names,
    domain_classes, domain_name, save_path,
):
    """
    Filter predictions to only samples belonging to domain_classes,
    then compute and print metrics + save heatmap for that domain.
    """
    # Indices of classes that belong to this domain
    domain_idx = {
        i for i, name in enumerate(class_names)
        if name in domain_classes
    }

    # Filter to only samples from this domain
    filtered = [
        (t, p) for t, p in zip(all_true, all_pred)
        if t in domain_idx
    ]

    if not filtered:
        print(f"  No samples found for {domain_name}. Skipping.")
        return

    d_true, d_pred = zip(*filtered)
    d_true = list(d_true)
    d_pred = list(d_pred)

    # Class names for this domain only (sorted by index)
    domain_class_names = [
        class_names[i] for i in sorted(domain_idx)
    ]
    # Re-map indices to 0-based for this domain
    idx_remap = {old: new for new, old in enumerate(sorted(domain_idx))}
    d_true_r  = [idx_remap[i] for i in d_true]
    d_pred_r  = [idx_remap[i] for i in d_pred]

    acc  = accuracy_score(d_true_r, d_pred_r)
    prec = precision_score(d_true_r, d_pred_r, average="weighted", zero_division=0)
    rec  = recall_score(d_true_r, d_pred_r, average="weighted", zero_division=0)
    f1   = f1_score(d_true_r, d_pred_r, average="weighted", zero_division=0)

    print(f"  Samples   : {len(d_true)}")
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%  (weighted)")
    print(f"  Recall    : {rec  * 100:.2f}%  (weighted)")
    print(f"  F1 Score  : {f1   * 100:.2f}%  (weighted)")

    print(f"\n  Per-Class Report ({domain_name}):")
    print(classification_report(
        d_true_r, d_pred_r,
        target_names=domain_class_names, zero_division=0,
    ))

    dm_cm = confusion_matrix(d_true_r, d_pred_r,
                             labels=list(range(len(domain_class_names))))
    save_heatmap(
        dm_cm, domain_class_names,
        f"{domain_name} — Confusion Matrix",
        save_path,
        figsize=(8, 6) if len(domain_class_names) <= 4 else (12, 10),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()
