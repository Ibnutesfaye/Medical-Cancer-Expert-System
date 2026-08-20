"""
Benchmark Routes
=================
GET  /benchmark/results       — list saved benchmark results
POST /benchmark/save          — save a benchmark result to DB
GET  /benchmark/compare       — latest comparison table
GET  /benchmark/history       — results over time
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from db.database import get_db
from core.security import get_current_user_payload
from models.benchmark_result import BenchmarkResult

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


class BenchmarkSaveRequest(BaseModel):
    experiment_name:     Optional[str] = None
    inference_mode:      str
    inference_time_ms:   Optional[int] = None
    encryption_time_ms:  Optional[int] = None
    decryption_time_ms:  Optional[int] = None
    memory_mb:           Optional[float] = None
    cpu_percent:         Optional[float] = None
    ciphertext_size_kb:  Optional[int] = None
    security_level_bits: Optional[int] = 0
    cancer_detected:     Optional[bool] = None
    cancer_type:         Optional[str] = None
    confidence:          Optional[float] = None
    model_name:          Optional[str] = None
    privacy_level:       Optional[str] = None
    error:               Optional[str] = None


@router.post("/save", status_code=201)
def save_benchmark(
    data: BenchmarkSaveRequest,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Save a benchmark result to the database."""
    record = BenchmarkResult(
        experiment_name     = data.experiment_name,
        inference_mode      = data.inference_mode,
        inference_time_ms   = data.inference_time_ms,
        encryption_time_ms  = data.encryption_time_ms,
        decryption_time_ms  = data.decryption_time_ms,
        memory_mb           = data.memory_mb,
        cpu_percent         = data.cpu_percent,
        ciphertext_size_kb  = data.ciphertext_size_kb,
        security_level_bits = data.security_level_bits or 0,
        cancer_detected     = data.cancer_detected,
        cancer_type         = data.cancer_type,
        confidence          = data.confidence,
        model_name          = data.model_name or "ResNet18",
        privacy_level       = data.privacy_level,
        run_by              = payload["user_id"],
        error               = data.error,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "message": "Benchmark result saved"}


@router.get("/results")
def list_results(
    skip: int = 0,
    limit: int = 50,
    mode: Optional[str] = None,
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """List benchmark results, optionally filtered by mode."""
    q = db.query(BenchmarkResult)
    if mode:
        q = q.filter(BenchmarkResult.inference_mode == mode)
    results = q.order_by(BenchmarkResult.created_at.desc()).offset(skip).limit(limit).all()
    return [
        {
            "id":                  r.id,
            "experiment_name":     r.experiment_name,
            "inference_mode":      r.inference_mode,
            "inference_time_ms":   r.inference_time_ms,
            "encryption_time_ms":  r.encryption_time_ms,
            "memory_mb":           r.memory_mb,
            "security_level_bits": r.security_level_bits,
            "cancer_detected":     r.cancer_detected,
            "cancer_type":         r.cancer_type,
            "confidence":          r.confidence,
            "privacy_level":       r.privacy_level,
            "error":               r.error,
            "created_at":          r.created_at.isoformat() if r.created_at else None,
        }
        for r in results
    ]


@router.get("/compare")
def compare_modes(
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """
    Return latest result per mode for side-by-side comparison.
    """
    modes = ["standard", "concrete_ml", "openfhe"]
    comparison = {}
    for mode in modes:
        latest = (
            db.query(BenchmarkResult)
            .filter(BenchmarkResult.inference_mode == mode,
                    BenchmarkResult.error == None)
            .order_by(BenchmarkResult.created_at.desc())
            .first()
        )
        if latest:
            comparison[mode] = {
                "inference_time_ms":   latest.inference_time_ms,
                "encryption_time_ms":  latest.encryption_time_ms,
                "memory_mb":           latest.memory_mb,
                "security_level_bits": latest.security_level_bits,
                "privacy_level":       latest.privacy_level,
                "confidence":          latest.confidence,
                "run_at":              latest.created_at.isoformat() if latest.created_at else None,
            }
        else:
            comparison[mode] = None
    return comparison


@router.get("/stats")
def benchmark_stats(
    payload: dict = Depends(get_current_user_payload),
    db: Session = Depends(get_db),
):
    """Average metrics per mode across all saved results."""
    from sqlalchemy import func as sqlfunc
    stats = {}
    modes = ["standard", "concrete_ml", "openfhe"]
    for mode in modes:
        row = (
            db.query(
                sqlfunc.avg(BenchmarkResult.inference_time_ms).label("avg_inference_ms"),
                sqlfunc.avg(BenchmarkResult.memory_mb).label("avg_memory_mb"),
                sqlfunc.count(BenchmarkResult.id).label("total_runs"),
            )
            .filter(BenchmarkResult.inference_mode == mode)
            .first()
        )
        stats[mode] = {
            "avg_inference_ms": round(row.avg_inference_ms or 0, 1),
            "avg_memory_mb":    round(row.avg_memory_mb or 0, 1),
            "total_runs":       row.total_runs or 0,
        }
    return stats
