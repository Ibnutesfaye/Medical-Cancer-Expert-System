"""
Concrete ML FHE Server
=======================
Runs encrypted inference on the server side.
Receives encrypted feature vectors + evaluation keys from the client,
performs FHE inference, and returns encrypted predictions.

The server NEVER decrypts — it only operates on ciphertexts.
"""

from __future__ import annotations

import json
import time
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

BASE       = Path(__file__).parent
MODEL_DIR  = BASE / "models"
FHE_MODEL_PATH = MODEL_DIR / "fhe_classifier"
METRICS_PATH   = MODEL_DIR / "fhe_metrics.json"

# Cancer class names (for binary label lookup)
CANCER_CLASSES = {
    "glioma_tumor", "meningioma_tumor", "pituitary_tumor",
    "malignant_lung_cancer", "melanoma",
    "basal_cell_carcinoma", "actinic_keratosis",
}


class FHEInferenceServer:
    """
    Server-side FHE inference engine.
    Supports both Concrete ML (real FHE) and sklearn fallback.
    """

    def __init__(self):
        self._fhe_server   = None
        self._sklearn_model = None
        self._use_fhe      = False
        self._loaded       = False
        self._metrics: dict = {}

    def load(self):
        if self._loaded:
            return

        # Try Concrete ML first
        try:
            from concrete.ml.deployment import FHEModelServer
            if FHE_MODEL_PATH.exists():
                self._fhe_server = FHEModelServer(path_dir=str(FHE_MODEL_PATH))
                self._fhe_server.load()
                self._use_fhe = True
                print("  [FHEServer] Concrete ML model loaded ✓")
            else:
                raise FileNotFoundError("FHE model not compiled yet")
        except Exception as e:
            print(f"  [FHEServer] Concrete ML unavailable: {e}")
            # Fallback to sklearn
            sklearn_path = MODEL_DIR / "sklearn_classifier.pkl"
            if sklearn_path.exists():
                with open(sklearn_path, "rb") as f:
                    # This fallback is a versioned, repository-controlled model,
                    # never an uploaded or request-provided pickle.
                    self._sklearn_model = pickle.load(f)  # nosec B301
                print("  [FHEServer] Using sklearn fallback ✓")
            else:
                print("  [FHEServer] No model found — run fhe_classifier.py first")

        # Load metrics
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                self._metrics = json.load(f)

        self._loaded = True

    def run_encrypted(
        self,
        encrypted_input: bytes,
        serialized_eval_keys: bytes,
    ) -> bytes:
        """
        Run FHE inference on encrypted input (Concrete ML mode).

        Args:
            encrypted_input:      Serialized encrypted feature vector
            serialized_eval_keys: Evaluation keys from client

        Returns:
            Serialized encrypted prediction
        """
        if not self._use_fhe or self._fhe_server is None:
            raise RuntimeError("FHE server not loaded. Run fhe_classifier.py first.")
        return self._fhe_server.run(encrypted_input, serialized_eval_keys)

    def run_plaintext(
        self,
        features: np.ndarray,
        idx_to_class: dict[int, str],
    ) -> dict:
        """
        Run plaintext inference using sklearn fallback.
        Used when Concrete ML is not installed or model not yet trained.

        Returns result dict compatible with standard image_analyzer output.
        """
        if self._sklearn_model is None:
            # No model trained yet — return a clear informational result
            return {
                "cancer_detected":   None,
                "cancer_type":       None,
                "confidence":        None,
                "inference_mode":    "concrete_ml",
                "inference_time_ms": 0,
                "encryption_time_ms": 0,
                "decryption_time_ms": 0,
                "security_level_bits": 0,
                "model_accuracy":    0.0,
                "message":           "FHE classifier not trained yet. Run: venv\\Scripts\\python.exe -m ai.concrete_ml.fhe_classifier",
                "status":            "not_trained",
                "safety_message":    "This result is based on trained dataset. Consult doctor.",
            }

        t0 = time.time()
        proba = self._sklearn_model.predict_proba(features.reshape(1, -1))[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        elapsed_ms = int((time.time() - t0) * 1000)

        cancer_type = idx_to_class.get(pred_idx, "unknown")
        is_cancer   = cancer_type in CANCER_CLASSES

        # Binary cancer probability = sum of all cancer-class probs
        cancer_prob = sum(
            float(proba[i])
            for i, name in idx_to_class.items()
            if name in CANCER_CLASSES and i < len(proba)
        )

        return {
            "cancer_detected":   is_cancer,
            "cancer_type":       cancer_type.replace("_", " ").title() if is_cancer else None,
            "confidence":        round(cancer_prob if is_cancer else 1.0 - cancer_prob, 3),
            "inference_mode":    "concrete_ml_fallback",
            "inference_time_ms": elapsed_ms,
            "encryption_time_ms": 0,
            "decryption_time_ms": 0,
            "security_level_bits": 0,
            "model_accuracy":    self._metrics.get("accuracy", 0.0),
            "message":           "Cancer detected (FHE fallback)" if is_cancer else "No cancer detected (FHE fallback)",
            "safety_message":    "This result is based on trained dataset. Consult doctor.",
        }

    @property
    def is_fhe_enabled(self) -> bool:
        return self._use_fhe

    @property
    def is_ready(self) -> bool:
        return self._use_fhe or self._sklearn_model is not None

    @property
    def metrics(self) -> dict:
        return self._metrics


# Singleton
_server: Optional[FHEInferenceServer] = None


def get_fhe_server() -> FHEInferenceServer:
    global _server
    if _server is None:
        _server = FHEInferenceServer()
        _server.load()
    return _server
