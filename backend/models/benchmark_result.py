"""
BenchmarkResult ORM Model
Stores results from all three inference mode comparisons.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.sql import func
from db.database import Base


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id                  = Column(Integer, primary_key=True, index=True)
    experiment_name     = Column(String(128), nullable=True)
    inference_mode      = Column(String(32), nullable=False)   # standard/concrete_ml/openfhe
    accuracy            = Column(Float, nullable=True)
    precision_score     = Column(Float, nullable=True)
    recall_score        = Column(Float, nullable=True)
    f1_score            = Column(Float, nullable=True)
    inference_time_ms   = Column(Integer, nullable=True)
    encryption_time_ms  = Column(Integer, nullable=True)
    decryption_time_ms  = Column(Integer, nullable=True)
    memory_mb           = Column(Float, nullable=True)
    cpu_percent         = Column(Float, nullable=True)
    ciphertext_size_kb  = Column(Integer, nullable=True)
    security_level_bits = Column(Integer, default=0)
    model_name          = Column(String(64), nullable=True)
    dataset             = Column(String(64), nullable=True)
    num_samples         = Column(Integer, nullable=True)
    cancer_detected     = Column(Boolean, nullable=True)
    cancer_type         = Column(String(128), nullable=True)
    confidence          = Column(Float, nullable=True)
    privacy_level       = Column(String(64), nullable=True)
    run_by              = Column(Integer, nullable=True)
    error               = Column(String(512), nullable=True)
    created_at          = Column(DateTime, server_default=func.now())
