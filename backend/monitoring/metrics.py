"""
Prometheus Metrics Exporter
============================
Exposes application metrics at GET /metrics for Prometheus scraping.
Grafana reads from Prometheus to display dashboards.

Metrics tracked:
  - inference_requests_total (by mode: standard/concrete_ml/openfhe)
  - inference_duration_seconds (histogram by mode)
  - cancer_detections_total (by type)
  - rag_queries_total (by source: document/wikipedia/pubmed/llm)
  - active_users (gauge)
  - model_accuracy (gauge by dataset)
"""

from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import (
        Counter, Histogram, Gauge,
        generate_latest, CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ── Metric definitions ────────────────────────────────────────────────────────

if PROMETHEUS_AVAILABLE:
    INFERENCE_REQUESTS = Counter(
        'inference_requests_total',
        'Total inference requests',
        ['mode'],          # standard / concrete_ml / openfhe
    )

    INFERENCE_DURATION = Histogram(
        'inference_duration_seconds',
        'Inference latency in seconds',
        ['mode'],
        buckets=[0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
    )

    CANCER_DETECTIONS = Counter(
        'cancer_detections_total',
        'Total cancer detections by type',
        ['cancer_type', 'mode'],
    )

    RAG_QUERIES = Counter(
        'rag_queries_total',
        'Total RAG chat queries by source',
        ['source'],        # document / wikipedia / pubmed / llm
    )

    ACTIVE_USERS = Gauge(
        'active_users',
        'Estimated active users (sessions in last 24h)',
    )

    MODEL_ACCURACY = Gauge(
        'model_accuracy',
        'Current model accuracy by dataset',
        ['dataset'],       # brain / lung / skin / overall
    )

    FHE_ENCRYPTION_DURATION = Histogram(
        'fhe_encryption_duration_seconds',
        'FHE encryption time',
        ['mode'],
        buckets=[0.1, 0.5, 1, 2, 5, 10],
    )


# ── Helper functions ──────────────────────────────────────────────────────────

def record_inference(mode: str, duration_s: float, cancer_detected: bool,
                     cancer_type: Optional[str] = None):
    """Record an inference request."""
    if not PROMETHEUS_AVAILABLE:
        return
    INFERENCE_REQUESTS.labels(mode=mode).inc()
    INFERENCE_DURATION.labels(mode=mode).observe(duration_s)
    if cancer_detected and cancer_type:
        CANCER_DETECTIONS.labels(
            cancer_type=cancer_type.lower().replace(" ", "_"),
            mode=mode,
        ).inc()


def record_rag_query(source: str):
    """Record a RAG query by source."""
    if not PROMETHEUS_AVAILABLE:
        return
    RAG_QUERIES.labels(source=source).inc()


def set_model_accuracy(dataset: str, accuracy: float):
    """Update model accuracy gauge."""
    if not PROMETHEUS_AVAILABLE:
        return
    MODEL_ACCURACY.labels(dataset=dataset).set(accuracy)


def get_metrics_response() -> tuple[bytes, str]:
    """Return Prometheus metrics as bytes + content type."""
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST
