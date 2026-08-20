"""
Encrypted Inference Routes
===========================
Endpoints for Concrete ML and OpenFHE inference modes.
All three modes share a common POST /inference/analyze endpoint
with a mode selector.

Routes:
  POST /inference/analyze          — run any inference mode
  GET  /inference/modes            — list available modes + status
  GET  /inference/{id}/gradcam     — get Grad-CAM for a result
  GET  /inference/benchmark/quick  — quick benchmark on test image
"""

from __future__ import annotations

import time
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from db.database import get_db
from core.security import get_current_user_payload
from image_analyzer import get_image_analyzer
from label_classes_loader import get_idx_to_class

router = APIRouter(prefix="/inference", tags=["encrypted-inference"])


class InferenceRequest(BaseModel):
    mode: str = "standard"   # "standard" | "concrete_ml" | "openfhe"


# ── Mode information ──────────────────────────────────────────────────────────

@router.get("/modes")
def list_modes():
    """Return available inference modes and their status."""
    # Check if FHE server is ready
    fhe_ready = False
    fhe_msg   = "Concrete ML model not compiled — run: python -m ai.concrete_ml.fhe_classifier"
    try:
        from ai.concrete_ml.fhe_server import get_fhe_server
        srv = get_fhe_server()
        fhe_ready = srv.is_ready
        if not fhe_ready:
            fhe_msg = "No classifier trained — run: python -m ai.concrete_ml.fhe_classifier"
        elif srv.is_fhe_enabled:
            fhe_msg = "FHE (Concrete ML compiled)"
        else:
            fhe_msg = "Sklearn fallback (install concrete-ml for real FHE)"
    except Exception as e:
        fhe_msg = str(e)

    return {
        "modes": [
            {
                "id":          "standard",
                "name":        "Standard AI",
                "description": "ResNet18 CNN — fast, highest accuracy, no privacy",
                "available":   True,
                "privacy":     "none",
                "latency_est": "~50ms",
                "security":    "0-bit",
            },
            {
                "id":          "concrete_ml",
                "name":        "Concrete ML (FHE)",
                "description": "Privacy-preserving inference using Fully Homomorphic Encryption",
                "available":   fhe_ready,
                "status_msg":  fhe_msg,
                "privacy":     "feature-level FHE",
                "latency_est": "~2–10s",
                "security":    "128-bit",
            },
            {
                "id":          "openfhe",
                "name":        "OpenFHE (CKKS)",
                "description": "Pixel-level FHE using CKKS scheme — research only",
                "available":   False,
                "status_msg":  "Planned — see RESEARCH_PLATFORM.md",
                "privacy":     "pixel-level FHE",
                "latency_est": "~2–5 minutes",
                "security":    "128-bit",
            },
        ]
    }


# ── Main inference endpoint ────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_with_mode(
    file: UploadFile = File(...),
    mode: str = "standard",
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """
    Analyze a medical image using the specified inference mode.

    Mode options:
      standard    — existing ResNet18 pipeline
      concrete_ml — privacy-preserving FHE inference
      openfhe     — homomorphic CKKS (placeholder)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size exceeds 20 MB")

    t_start = time.perf_counter()

    if mode == "standard":
        analyzer = get_image_analyzer()
        result   = analyzer.analyze(content)
        result["inference_mode"]    = "standard"
        result["inference_time_ms"] = int((time.perf_counter() - t_start) * 1000)

    elif mode == "concrete_ml":
        try:
            from ai.concrete_ml.feature_extractor import get_feature_extractor
            from ai.concrete_ml.fhe_server import get_fhe_server

            extractor    = get_feature_extractor()
            server       = get_fhe_server()
            idx_to_class = get_idx_to_class()

            features = extractor.extract(content)
            result   = server.run_plaintext(features, idx_to_class)
            result["inference_mode"]    = "concrete_ml"
            result["inference_time_ms"] = int((time.perf_counter() - t_start) * 1000)
        except Exception as e:
            raise HTTPException(status_code=503,
                detail=f"Concrete ML inference failed: {e}")

    elif mode == "openfhe":
        result = {
            "inference_mode":    "openfhe",
            "cancer_detected":   None,
            "cancer_type":       None,
            "confidence":        None,
            "message":           "OpenFHE CKKS inference is planned — see RESEARCH_PLATFORM.md",
            "status":            "not_implemented",
            "estimated_time_s":  120,
            "inference_time_ms": 0,
        }
    else:
        raise HTTPException(status_code=400,
            detail=f"Unknown mode '{mode}'. Use: standard, concrete_ml, openfhe")

    return result


# ── Grad-CAM endpoint ──────────────────────────────────────────────────────────

@router.post("/gradcam")
async def get_gradcam(
    file: UploadFile = File(...),
    class_idx: Optional[int] = None,
    payload: dict = Depends(get_current_user_payload),
):
    """Generate Grad-CAM heatmap overlay for an image."""
    content = await file.read()

    try:
        import torch
        from image_analyzer import get_image_analyzer
        from ai.explainability.gradcam import GradCAMGenerator

        analyzer = get_image_analyzer()
        # Get the underlying model
        model = analyzer._predictor._model if hasattr(analyzer, '_predictor') else None
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        gen = GradCAMGenerator(model)
        b64_overlay, pred_class, key_regions = gen.generate_b64(content, class_idx)

        from label_classes_loader import get_idx_to_class
        idx_to_class = get_idx_to_class()

        return {
            "heatmap_b64":  b64_overlay,
            "predicted_class_idx":  pred_class,
            "predicted_class_name": idx_to_class.get(pred_class, "unknown"),
            "key_regions":  key_regions,
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="cv2 (opencv) not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Quick benchmark endpoint ───────────────────────────────────────────────────

@router.post("/benchmark/quick")
async def quick_benchmark(
    file: UploadFile = File(...),
    payload: dict = Depends(get_current_user_payload),
):
    """Run a quick benchmark comparing Standard AI vs Concrete ML on one image."""
    content = await file.read()

    from ai.benchmark.benchmark_service import get_benchmark_service
    svc    = get_benchmark_service()
    result = svc.run_full_benchmark(content)
    return result
