"""
AI Cancer Detection — Trained Model Prediction

Pipeline:
  Step 1 → receive image
  Step 2 → send to trained ResNet18 model (trained on dataset.csv + images)
  Step 3 → model predicts cancer type
  Step 4 → map prediction to cancer/non-cancer label
  Step 5 → return result

If model not trained yet, falls back to ResNet18 embedding similarity.
Train first with: venv/Scripts/python.exe train_model.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageStat
import torchvision.models as models
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE        = Path(__file__).parent
CSV_PATH    = BASE / "dataset.csv"
MODEL_PATH  = BASE / "cancer_model.pth"
LABELS_PATH = BASE / "label_classes.json"
SAFETY      = "This result is based on trained dataset. Consult doctor."

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

# Minimum confidence to return a prediction (below this → low-confidence rejection)
# With 14 classes, a correct prediction at 0.30+ is meaningful.
# Skin dermoscopy images have lower per-class confidence due to visual similarity.
CONFIDENCE_THRESHOLD = 0.30

# HSV saturation threshold for medical image validation.
# Brain MRI / Lung CT: saturation ≈ 0–20 (near greyscale)
# Skin dermoscopy:     saturation ≈ 40–140 (colour images)
# Natural photos:      saturation ≈ 80–200+ (vivid colours)
# Set high enough to allow all three medical modalities through,
# while still rejecting obviously non-medical images (cartoons, objects, etc.)
MAX_COLOR_SATURATION_MEAN = 160  # allows brain, lung AND skin images

UNKNOWN_IMAGE_MSG = "Unknown image. This image type was not included in the training dataset."
LOW_CONFIDENCE_MSG = "The model is not confident enough to make a reliable prediction."

# Image dirs for fallback similarity search
IMAGE_DIRS = [
    BASE / "brain cancer csv" / "Testing",
    BASE / "brain cancer csv" / "Training",
    BASE / "lung cancer csv" / "The IQ-OTHNCCD lung cancer dataset"
         / "The IQ-OTHNCCD lung cancer dataset",
    BASE / "skin cancer csv" / "HAM10000_images_part_1",
    BASE / "skin cancer csv" / "HAM10000_images_part_2",
]

# Skin cancer malignant dx codes
SKIN_CANCER_DX = {"mel", "bcc", "akiec"}
# Brain cancer malignant folder names
BRAIN_CANCER_FOLDERS = {"glioma", "meningioma", "pituitary"}

# Transform for inference — must match VAL_TRANSFORM used during training exactly.
# Training used NO Grayscale step, so inference must not use it either.
# Applying Grayscale here would destroy colour information that skin cancer
# detection depends on, causing low confidence on dermoscopy images.
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _err(msg: str) -> dict:
    return {
        "cancer_detected": False,
        "cancer_type": None,
        "confidence": 0.0,
        "message": msg,
        "safety_message": SAFETY,
    }


def _build_file_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in IMAGE_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[p.name] = p
    return index


def _is_valid_medical_image(image: Image.Image) -> bool:
    """
    Lightweight heuristic to reject obviously non-medical images
    (natural photos, cartoons, objects, animals, etc.).

    Medical images (MRI, CT, dermoscopy) share these traits:
      - Low colour saturation (mostly greyscale or near-greyscale)
      - Relatively uniform background (dark or white)
      - No extreme RGB channel imbalance typical of natural photos

    Returns True if the image *could* be a medical scan, False otherwise.
    """
    # Convert to HSV and check mean saturation
    hsv = image.convert("HSV")
    stat = ImageStat.Stat(hsv)
    # stat.mean[1] is the S (saturation) channel mean (0–255)
    saturation_mean = stat.mean[1]
    if saturation_mean > MAX_COLOR_SATURATION_MEAN:
        return False

    # Additional check: if the image is very small it's likely an icon/cartoon
    w, h = image.size
    if w < 32 or h < 32:
        return False

    return True


# ---------------------------------------------------------------------------
# Trained model predictor
# ---------------------------------------------------------------------------

class _TrainedPredictor:
    """Uses the fine-tuned ResNet18 saved by train_model.py."""

    def __init__(self):
        # The checkpoint contains tensors and primitive metadata only. Restricting
        # unpickling prevents arbitrary Python objects from being constructed.
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        num_classes = checkpoint["num_classes"]
        self.class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
        self.idx_to_class: dict[int, str] = {v: k for k, v in self.class_to_idx.items()}
        
        self.val_acc = checkpoint.get("val_acc", 0.0)
        self.train_acc = checkpoint.get("train_acc", 0.0)
        self.train_loss = checkpoint.get("train_loss", 0.0)

        # Load label info (cancer vs non-cancer per class)
        if LABELS_PATH.exists():
            with open(LABELS_PATH) as f:
                info = json.load(f)
            self.cancer_labels: dict[str, str] = info.get("cancer_labels", {})
        else:
            # Fallback: infer from class name
            self.cancer_labels = {}

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._model = model
        print(f"  Trained model loaded (val_acc={checkpoint.get('val_acc', '?'):.3f}, "
              f"{num_classes} classes)")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        t = _INFER_TRANSFORM(image).unsqueeze(0)
        logits = self._model(t)                        # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0)   # (num_classes,)

        top_idx  = int(probs.argmax())
        top_prob = float(probs[top_idx])
        cancer_type = self.idx_to_class[top_idx]

        # Reject low-confidence predictions
        if top_prob < CONFIDENCE_THRESHOLD:
            return {
                "cancer_detected": False,
                "cancer_type": None,
                "confidence": round(top_prob, 3),
                "message": LOW_CONFIDENCE_MSG,
                "unknown_image": False,
                "low_confidence": True,
                "safety_message": SAFETY,
                "match_source": "trained_model",
                "model_used": "ResNet18 CNN",
            }

        # Determine cancer/non-cancer label
        label = self.cancer_labels.get(cancer_type, "")
        if not label:
            # Infer from name
            non_cancer_keywords = {"no_tumor", "normal", "benign", "nevus",
                                   "melanocytic_nevi", "dermatofibroma",
                                   "benign_keratosis", "vascular_lesion"}
            label = "non-cancer" if cancer_type.lower() in non_cancer_keywords else "cancer"

        # Also compute binary cancer probability (sum of all cancer-class probs)
        cancer_prob = sum(
            float(probs[idx])
            for cls, idx in self.class_to_idx.items()
            if self.cancer_labels.get(cls, "cancer") == "cancer"
        )

        # Information to include for transparency
        evaluation_info = {
            "decision_process": "Predictions are based on softmax probability outputs. The highest probability class is selected as output.",
            "dependency": "Results depend on learned features from training data.",
            "note": "The model is NOT clinically approved and is for educational/research purposes only."
        }

        if label == "cancer":
            return {
                "cancer_detected": True,
                "cancer_type": cancer_type.replace("_", " ").title(),
                "confidence": round(cancer_prob, 3),
                "message": "Cancer detected",
                "unknown_image": False,
                "low_confidence": False,
                "match_source": "trained_model",
                "safety_message": SAFETY,
                "training_accuracy": self.train_acc,
                "validation_accuracy": self.val_acc,
                "training_loss": self.train_loss,
                "model_used": "ResNet18 CNN",
                "evaluation_info": evaluation_info,
            }
        else:
            return {
                "cancer_detected": False,
                "cancer_type": None,
                "confidence": round(1.0 - cancer_prob, 3),
                "message": "No cancer detected",
                "unknown_image": False,
                "low_confidence": False,
                "match_source": "trained_model",
                "safety_message": SAFETY,
                "training_accuracy": self.train_acc,
                "validation_accuracy": self.val_acc,
                "training_loss": self.train_loss,
                "model_used": "ResNet18 CNN",
                "evaluation_info": evaluation_info,
            }


# ---------------------------------------------------------------------------
# Fallback: ResNet18 embedding similarity (used before training)
# ---------------------------------------------------------------------------

class _SimilarityFallback:
    """
    Fallback when no trained model exists yet.
    Uses pretrained ResNet18 features + cosine similarity against dataset images.
    Labels come from folder structure and HAM10000_metadata.csv.
    """

    def __init__(self):
        # Feature extractor (no classifier head)
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._extractor = nn.Sequential(*list(base.children())[:-1])
        self._extractor.eval()
        self._embeddings: Optional[torch.Tensor] = None
        self._entries: list[dict] = []
        self._built = False

    @torch.no_grad()
    def _embed(self, img: Image.Image) -> torch.Tensor:
        t = _INFER_TRANSFORM(img).unsqueeze(0)
        feat = self._extractor(t).squeeze()
        return F.normalize(feat, dim=0)

    def build_index(self):
        if self._built:
            return
        print("  Building similarity index (fallback, 50/class cap)...")
        t0 = time.time()

        # Collect entries per class, capped at MAX_PER_CLASS for speed
        MAX_PER_CLASS = 50
        buckets: dict[str, list[dict]] = {}

        def _add(path, cancer_type, label, source):
            key = cancer_type
            if key not in buckets:
                buckets[key] = []
            if len(buckets[key]) < MAX_PER_CLASS:
                buckets[key].append({"path": path, "cancer_type": cancer_type,
                                     "label": label, "source": source})

        # Brain
        for split in ("Testing", "Training"):
            root = BASE / "brain cancer csv" / split
            if not root.exists():
                continue
            for folder in root.iterdir():
                if not folder.is_dir():
                    continue
                fname = folder.name.lower()
                label = "cancer" if fname in BRAIN_CANCER_FOLDERS else "non-cancer"
                for p in list(folder.glob("*.jpg")) + list(folder.glob("*.png")):
                    _add(p, fname, label, "brain")

        # Lung
        lung_root = (BASE / "lung cancer csv"
                     / "The IQ-OTHNCCD lung cancer dataset"
                     / "The IQ-OTHNCCD lung cancer dataset")
        if lung_root.exists():
            for folder in lung_root.iterdir():
                if not folder.is_dir():
                    continue
                n = folder.name.lower()
                if "malignant" in n:
                    label, ct = "cancer", "lung_cancer"
                elif "benign" in n or "bengin" in n:
                    label, ct = "non-cancer", "benign_lung"
                else:
                    label, ct = "non-cancer", "normal_lung"
                for p in list(folder.glob("*.jpg")) + list(folder.glob("*.png")):
                    _add(p, ct, label, "lung")

        # Skin
        meta_csv = BASE / "skin cancer csv" / "HAM10000_metadata.csv"
        if meta_csv.exists():
            meta = pd.read_csv(meta_csv)
            skin_map = {
                str(r["image_id"]).strip(): (
                    str(r["dx"]).strip().lower(),
                    "cancer" if str(r["dx"]).strip().lower() in SKIN_CANCER_DX else "non-cancer"
                )
                for _, r in meta.iterrows()
            }
            for skin_dir in [BASE / "skin cancer csv" / "HAM10000_images_part_1",
                             BASE / "skin cancer csv" / "HAM10000_images_part_2"]:
                if not skin_dir.exists():
                    continue
                for p in skin_dir.glob("*.jpg"):
                    img_id = p.stem
                    if img_id in skin_map:
                        dx, lbl = skin_map[img_id]
                        _add(p, dx, lbl, "skin")

        entries = [e for bucket in buckets.values() for e in bucket]
        print(f"  {len(entries)} images sampled ({len(buckets)} classes), embedding...")

        vecs: list[torch.Tensor] = []
        valid: list[dict] = []
        for entry in entries:
            try:
                with Image.open(entry["path"]) as img:
                    v = self._embed(img.copy())
                vecs.append(v)
                valid.append(entry)
            except Exception:
                continue

        self._entries = valid
        self._embeddings = torch.stack(vecs) if vecs else None
        self._built = True
        print(f"  Fallback index ready: {len(valid)} images ({time.time()-t0:.1f}s)")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        self.build_index()
        if self._embeddings is None or not self._entries:
            return _err("Dataset not loaded.")

        q = self._embed(image)
        sims = (self._embeddings @ q).squeeze()
        k = min(5, len(self._entries))
        top_k = torch.topk(sims, k)

        cancer_score, non_cancer_score = 0.0, 0.0
        cancer_types: dict[str, float] = {}
        best_source = self._entries[top_k.indices[0].item()]["source"]

        for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist()):
            entry = self._entries[idx]
            w = max(float(score), 0.0)
            if entry["label"] == "cancer":
                cancer_score += w
                ct = entry["cancer_type"]
                cancer_types[ct] = cancer_types.get(ct, 0.0) + w
            else:
                non_cancer_score += w

        total = cancer_score + non_cancer_score
        conf = round(cancer_score / total, 3) if total > 0 else 0.5

        if cancer_score > non_cancer_score:
            best_ct = max(cancer_types, key=lambda k: cancer_types[k])
            return {
                "cancer_detected": True,
                "cancer_type": best_ct.replace("_", " ").title(),
                "confidence": conf,
                "message": "Cancer detected",
                "unknown_image": False,
                "low_confidence": False,
                "match_source": best_source,
                "safety_message": SAFETY,
            }
        else:
            return {
                "cancer_detected": False,
                "cancer_type": None,
                "confidence": round(1.0 - conf, 3),
                "message": "No cancer detected",
                "unknown_image": False,
                "low_confidence": False,
                "match_source": best_source,
                "safety_message": SAFETY,
            }


# ---------------------------------------------------------------------------
# ImageAnalyzer — uses trained model if available, else fallback
# ---------------------------------------------------------------------------

class ImageAnalyzer:

    def __init__(self):
        self._predictor = None
        self._built = False

    def _build_index(self):
        """Called on startup to initialize the predictor."""
        if self._built:
            return
        if MODEL_PATH.exists():
            try:
                print("Loading trained cancer model...")
                self._predictor = _TrainedPredictor()
                print("  Using trained model for predictions.")
            except Exception as e:
                print(f"  Failed to load trained model: {e}. Using fallback.")
                self._predictor = _SimilarityFallback()
                self._predictor.build_index()
        else:
            print("No trained model found. Using similarity fallback.")
            print("  Run: venv/Scripts/python.exe train_model.py  to train.")
            self._predictor = _SimilarityFallback()
            self._predictor.build_index()
        self._built = True

    def analyze(self, image_bytes: bytes) -> dict:
        """
        Step 1 → receive image
        Step 2 → validate it looks like a medical image
        Step 3 → send to trained model (or fallback)
        Step 4 → model predicts cancer type
        Step 5 → return result
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return _err(f"Could not read image: {e}")

        # Reject non-medical images before running the model
        if not _is_valid_medical_image(image):
            return {
                "cancer_detected": False,
                "cancer_type": None,
                "confidence": 0.0,
                "message": UNKNOWN_IMAGE_MSG,
                "unknown_image": True,
                "low_confidence": False,
                "safety_message": SAFETY,
            }

        # If index is still being built in background, build it now (blocking)
        if not self._built:
            self._build_index()

        if self._predictor is None:
            return _err("Predictor not initialized.")

        result = self._predictor.predict(image)

        # Apply confidence threshold for fallback predictor
        # (trained predictor already handles this internally)
        if not result.get("unknown_image") and not result.get("low_confidence"):
            conf = result.get("confidence", 0.0)
            if conf < CONFIDENCE_THRESHOLD:
                return {
                    "cancer_detected": False,
                    "cancer_type": None,
                    "confidence": round(conf, 3),
                    "message": LOW_CONFIDENCE_MSG,
                    "unknown_image": False,
                    "low_confidence": True,
                    "safety_message": SAFETY,
                }

        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_analyzer: Optional[ImageAnalyzer] = None


def get_image_analyzer() -> ImageAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ImageAnalyzer()
    return _analyzer
