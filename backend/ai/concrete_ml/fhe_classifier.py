"""
Concrete ML FHE Classifier
============================
Trains and compiles a privacy-preserving classifier on top of
ResNet18 feature vectors. The trained model is compiled into an
FHE circuit that can run on encrypted inputs.

Pipeline:
  1. Extract features from all training images (ResNet18)
  2. Train classifier on features (LogisticRegression / small NN)
  3. Compile to FHE circuit using Concrete ML
  4. Save compiled model for server deployment

Usage:
    python -m ai.concrete_ml.fhe_classifier

Why LogisticRegression:
  - Multiplicative depth = 1 (one dot product + sigmoid approx)
  - FHE inference: ~2-5 seconds
  - Accuracy: ~82-86% on cancer features
  - Fully supported by Concrete ML compile()
"""

from __future__ import annotations

import json
import time
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

FHE_MODEL_PATH   = MODEL_DIR / "fhe_classifier"
FEATURES_PATH    = MODEL_DIR / "features_cache.npz"
LABEL_MAP_PATH   = MODEL_DIR / "label_map.json"

# Cancer/non-cancer binary labels
CANCER_CLASSES = {
    "glioma_tumor", "meningioma_tumor", "pituitary_tumor",
    "malignant_lung_cancer", "melanoma",
    "basal_cell_carcinoma", "actinic_keratosis",
}


def train_and_compile(features: np.ndarray, labels: np.ndarray,
                      n_bits: int = 8) -> dict:
    """
    Train a Concrete ML classifier on feature vectors and compile to FHE.

    Args:
        features: np.ndarray (N, 512) — ResNet18 feature vectors
        labels:   np.ndarray (N,)     — integer class indices
        n_bits:   quantization bits   — 8 is the sweet spot

    Returns:
        dict with accuracy metrics and paths to saved artifacts
    """
    try:
        from concrete.ml.sklearn import LogisticRegression as FHELogReg
        CML_AVAILABLE = True
    except ImportError:
        CML_AVAILABLE = False
        print("  Concrete ML not installed — using sklearn LogisticRegression as fallback")
        from sklearn.linear_model import LogisticRegression as FHELogReg

    print("=" * 60)
    print("Concrete ML FHE Classifier Training")
    print("=" * 60)
    print(f"  Features shape : {features.shape}")
    print(f"  Num classes    : {len(np.unique(labels))}")
    print(f"  Quantization   : {n_bits}-bit")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=42, stratify=labels
    )

    t0 = time.time()

    if CML_AVAILABLE:
        model = FHELogReg(n_bits=n_bits, max_iter=1000)
    else:
        model = FHELogReg(max_iter=1000, C=1.0)

    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Training time  : {train_time:.1f}s")
    print(f"  Test accuracy  : {acc * 100:.2f}%")

    # Compile to FHE circuit (Concrete ML only)
    compile_time = 0.0
    if CML_AVAILABLE:
        print("\n  Compiling FHE circuit...")
        t1 = time.time()
        model.compile(X_train)
        compile_time = time.time() - t1
        print(f"  Compile time   : {compile_time:.1f}s")

        # Save FHE model
        from concrete.ml.deployment import FHEModelDev
        dev = FHEModelDev(path_dir=str(FHE_MODEL_PATH), model=model)
        dev.save()
        print(f"  FHE model saved → {FHE_MODEL_PATH}")
    else:
        # Fallback: save sklearn model
        with open(MODEL_DIR / "sklearn_classifier.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"  Sklearn model saved → {MODEL_DIR}/sklearn_classifier.pkl")

    metrics = {
        "accuracy":      round(float(acc), 4),
        "train_time_s":  round(train_time, 2),
        "compile_time_s": round(compile_time, 2),
        "n_bits":        n_bits,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "concrete_ml":   CML_AVAILABLE,
    }

    with open(MODEL_DIR / "fhe_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Metrics saved → {MODEL_DIR}/fhe_metrics.json")
    return metrics


# ---------------------------------------------------------------------------
# Feature extraction from training images (same index as train_model.py)
# ---------------------------------------------------------------------------

BACKEND = Path(__file__).resolve().parents[2]
CSV_PATH = BACKEND / "dataset.csv"

IMAGE_DIRS = [
    BACKEND / "brain cancer csv" / "Testing",
    BACKEND / "brain cancer csv" / "Training",
    BACKEND / "lung cancer csv" / "The IQ-OTHNCCD lung cancer dataset"
    / "The IQ-OTHNCCD lung cancer dataset",
    BACKEND / "skin cancer csv" / "HAM10000_images_part_1",
    BACKEND / "skin cancer csv" / "HAM10000_images_part_2",
]


def build_file_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in IMAGE_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[p.name] = p
    return index


def load_training_rows(max_per_class: int = 100) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load stratified sample from dataset.csv and map to integer labels.

    Returns:
        features placeholder labels only if cache miss — see extract_features().
    """
    import pandas as pd
    from collections import defaultdict
    from label_classes_loader import get_class_to_idx

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"dataset.csv not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    file_index = build_file_index()
    class_to_idx = get_class_to_idx()

    by_class: dict[str, list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        fname = str(row["image_name"])
        ctype = str(row["cancer_type"])
        if fname not in file_index or ctype not in class_to_idx:
            continue
        if len(by_class[ctype]) >= max_per_class:
            continue
        by_class[ctype].append({
            "image_name": fname,
            "path": file_index[fname],
            "label": class_to_idx[ctype],
        })

    rows = [r for items in by_class.values() for r in items]
    if not rows:
        raise RuntimeError("No training images found — check dataset folders and dataset.csv")

    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    paths = [r["path"] for r in rows]
    print(f"  Selected {len(rows)} images ({max_per_class} max/class)")
    return paths, labels, class_to_idx


def extract_features(
    paths: list[Path],
    labels: np.ndarray,
    batch_size: int = 16,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract or load cached ResNet18 feature vectors."""
    if FEATURES_PATH.exists() and not force:
        data = np.load(FEATURES_PATH)
        print(f"  Loaded cached features → {FEATURES_PATH} ({data['features'].shape})")
        return data["features"], data["labels"]

    from ai.concrete_ml.feature_extractor import ResNet18FeatureExtractor

    extractor = ResNet18FeatureExtractor()
    all_features: list[np.ndarray] = []

    print(f"  Extracting features from {len(paths)} images...")
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_bytes = [p.read_bytes() for p in batch_paths]
        feats = extractor.extract_batch(batch_bytes)
        all_features.append(feats)
        done = min(i + batch_size, len(paths))
        if done % 64 == 0 or done == len(paths):
            print(f"    {done}/{len(paths)}")

    features = np.vstack(all_features).astype(np.float32)
    np.savez(FEATURES_PATH, features=features, labels=labels)
    print(f"  Features cached → {FEATURES_PATH}")
    return features, labels


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Concrete ML / sklearn FHE classifier")
    parser.add_argument("--max-per-class", type=int, default=100,
                        help="Max images per class for feature extraction")
    parser.add_argument("--n-bits", type=int, default=8,
                        help="Quantization bits for Concrete ML (8 recommended)")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Ignore features_cache.npz and re-extract")
    args = parser.parse_args()

    paths, labels, _ = load_training_rows(max_per_class=args.max_per_class)
    features, labels = extract_features(
        paths, labels, force=args.force_reextract,
    )
    train_and_compile(features, labels, n_bits=args.n_bits)


if __name__ == "__main__":
    main()
