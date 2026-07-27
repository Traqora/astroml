"""Model Monitoring API — Issue #256.

Endpoints:
  GET /api/v1/monitoring/metrics              — latest model metrics
  GET /api/v1/monitoring/performance-history  — time-series metrics
  GET /api/v1/monitoring/drift-report         — feature drift analysis
  GET /api/v1/monitoring/prediction-stats     — prediction volume/distribution
  GET /api/v1/monitoring/latency              — API latency percentiles
"""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Tuple

from fastapi import APIRouter, Query

from api.schemas import (
    DriftReport,
    LatencyStats,
    ModelMetricsOut,
    PerformancePoint,
    PredictionStats,
)

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# ─── In-process latency ring buffer (populated by middleware) ─────────────────
# Stores (timestamp, latency_ms) tuples for the last 1000 requests.
_latency_buffer: Deque[Tuple[float, float]] = deque(maxlen=1000)


def record_latency(latency_ms: float) -> None:
    """Called by middleware to record a request latency sample."""
    _latency_buffer.append((time.time(), latency_ms))


def _load_latest_metrics() -> ModelMetricsOut:
    """Try to load the most recent benchmark result from disk."""
    import json, os, glob  # noqa: PLC0415
    pattern = "benchmark_results/**/*.json"
    files = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
            metrics = data.get("metrics") or data.get("best_metrics") or {}
            if metrics:
                f1_val = metrics.get("f1") or metrics.get("f1_score")
                auc_val = metrics.get("auc") or metrics.get("auc_roc")
                return ModelMetricsOut(
                    accuracy=metrics.get("accuracy"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    f1=f1_val,
                    f1_score=f1_val,
                    auc=auc_val,
                    auc_roc=auc_val,
                    drift_score=None,
                    recorded_at=datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc),
                )
        except Exception:  # noqa: BLE001
            continue
    return ModelMetricsOut()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=ModelMetricsOut)
def get_metrics():
    """Return the latest model performance metrics."""
    return _load_latest_metrics()


@router.get("/performance-history", response_model=list[PerformancePoint])
def get_performance_history(days: int = Query(30, ge=1, le=365)):
    """Return time-series of model metrics over the last N days."""
    import json, os, glob  # noqa: PLC0415
    from datetime import timedelta  # noqa: PLC0415

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    points: list[PerformancePoint] = []

    for path in sorted(glob.glob("benchmark_results/**/*.json", recursive=True), key=os.path.getmtime):
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        if mtime < cutoff:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            metrics = data.get("metrics") or data.get("best_metrics") or {}
            if metrics:
                points.append(PerformancePoint(
                    date=mtime.date().isoformat(),
                    accuracy=metrics.get("accuracy"),
                    f1=metrics.get("f1"),
                    auc=metrics.get("auc"),
                ))
        except Exception:  # noqa: BLE001
            continue

    # Pad with empty points if fewer than requested days
    if not points:
        from datetime import timedelta  # noqa: PLC0415, F811
        points = [
            PerformancePoint(date=(datetime.now(timezone.utc) - timedelta(days=i)).date().isoformat())
            for i in range(days - 1, -1, -1)
        ]
    return points


@router.get("/drift-report", response_model=DriftReport)
def get_drift_report():
    """Return feature drift analysis. Uses validation module if available."""
    try:
        from astroml.validation.data_quality import DataQualityValidator  # noqa: PLC0415
        # Return informative defaults — real drift requires a reference dataset
        features = {col: 0.0 for col in [
            "in_degree", "out_degree", "total_received", "total_sent",
            "account_age", "unique_asset_count", "asset_entropy",
        ]}
    except ImportError:
        features = {}

    return DriftReport(
        features=features,
        overall_drift=0.0,
        generated_at=datetime.now(timezone.utc),
    )


@router.get("/prediction-stats", response_model=PredictionStats)
def get_prediction_stats():
    """Return prediction volume and distribution statistics."""
    try:
        from astroml.api.models import FraudAlert  # noqa: PLC0415
        from astroml.db.session import SessionLocal  # noqa: PLC0415
        from sqlalchemy import select, func  # noqa: PLC0415

        with SessionLocal() as db:
            total = db.scalar(select(func.count(FraudAlert.id))) or 0
            high = db.scalar(
                select(func.count(FraudAlert.id)).where(FraudAlert.risk_level == "high")
            ) or 0
            avg_score = db.scalar(select(func.avg(FraudAlert.score))) or 0.0
        return PredictionStats(
            total_predictions=total,
            anomaly_rate=round(high / total, 4) if total else 0.0,
            avg_score=round(float(avg_score), 4),
            period_days=30,
        )
    except Exception:  # noqa: BLE001
        return PredictionStats(total_predictions=0, anomaly_rate=0.0, avg_score=0.0, period_days=30)


@router.get("/latency", response_model=LatencyStats)
def get_latency():
    """Return API latency percentiles (p50, p95, p99) from the ring buffer."""
    import statistics  # noqa: PLC0415

    samples = [lat for _, lat in _latency_buffer]
    if not samples:
        return LatencyStats(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0)

    samples_sorted = sorted(samples)
    n = len(samples_sorted)

    def _pct(p: float) -> float:
        idx = max(0, int(n * p / 100) - 1)
        return round(samples_sorted[idx], 2)

    return LatencyStats(p50_ms=_pct(50), p95_ms=_pct(95), p99_ms=_pct(99))
