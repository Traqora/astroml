"""Model extraction (model stealing) detection.

Resolves part of #645.

Model extraction attacks probe a prediction API with many synthetic or
boundary-hugging queries in order to train a surrogate copy of the model.  This
module tracks per-client query behaviour and scores it against the signatures
that distinguish extraction from legitimate traffic:

* **Volume** — query rate far above the population norm.
* **Coverage** — inputs that sweep the feature space uniformly rather than
  clustering the way real traffic does.
* **Boundary probing** — a high share of low-confidence predictions.
* **Near-duplicates** — repeated queries differing in one feature, the
  signature of finite-difference gradient estimation.

Scores are advisory: feed them into rate limiting or step-up authentication
rather than hard-blocking on them alone.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ExtractionRisk",
    "ExtractionSignal",
    "ExtractionVerdict",
    "ModelExtractionDetector",
    "QueryRecord",
]


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ExtractionRisk(str, Enum):
    """Overall risk band assigned to a client."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExtractionSignal(str, Enum):
    """Individual behavioural signals that contribute to the risk score."""

    HIGH_VOLUME = "high_volume"
    UNIFORM_COVERAGE = "uniform_coverage"
    BOUNDARY_PROBING = "boundary_probing"
    NEAR_DUPLICATE_PROBING = "near_duplicate_probing"


@dataclass(frozen=True)
class QueryRecord:
    """A single scoring request observed on the prediction API."""

    client_id: str
    features: NDArray[np.float64]
    confidence: float
    timestamp: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        """Validate the confidence value."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


@dataclass(frozen=True)
class ExtractionVerdict:
    """Risk assessment for one client."""

    client_id: str
    risk: ExtractionRisk
    score: float
    signals: tuple[ExtractionSignal, ...]
    query_count: int
    queries_per_minute: float
    mean_confidence: float
    coverage_uniformity: float
    near_duplicate_ratio: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the verdict."""
        return {
            "client_id": self.client_id,
            "risk": self.risk.value,
            "score": self.score,
            "signals": [signal.value for signal in self.signals],
            "query_count": self.query_count,
            "queries_per_minute": self.queries_per_minute,
            "mean_confidence": self.mean_confidence,
            "coverage_uniformity": self.coverage_uniformity,
            "near_duplicate_ratio": self.near_duplicate_ratio,
            "details": dict(self.details),
        }


class ModelExtractionDetector:
    """Scores per-client query behaviour for model-extraction risk.

    Parameters
    ----------
    window:
        Sliding window over which behaviour is assessed.
    max_queries_per_minute:
        Rate above which the volume signal fires.
    low_confidence_threshold:
        Predictions below this confidence count as boundary probes.
    boundary_probe_ratio:
        Share of boundary probes above which that signal fires.
    near_duplicate_epsilon:
        L-infinity distance under which two queries count as near-duplicates.
    min_queries:
        Minimum observations before a client is scored at all.
    """

    def __init__(
        self,
        *,
        window: timedelta = timedelta(minutes=10),
        max_queries_per_minute: float = 60.0,
        low_confidence_threshold: float = 0.6,
        boundary_probe_ratio: float = 0.4,
        near_duplicate_epsilon: float = 1e-3,
        near_duplicate_ratio: float = 0.3,
        uniformity_threshold: float = 0.8,
        min_queries: int = 20,
        max_history_per_client: int = 5_000,
    ) -> None:
        if window.total_seconds() <= 0:
            raise ValueError("window must be positive")
        if min_queries < 2:
            raise ValueError("min_queries must be at least 2")
        self.window = window
        self.max_queries_per_minute = max_queries_per_minute
        self.low_confidence_threshold = low_confidence_threshold
        self.boundary_probe_ratio = boundary_probe_ratio
        self.near_duplicate_epsilon = near_duplicate_epsilon
        self.near_duplicate_ratio = near_duplicate_ratio
        self.uniformity_threshold = uniformity_threshold
        self.min_queries = min_queries
        self._history: defaultdict[str, deque[QueryRecord]] = defaultdict(
            lambda: deque(maxlen=max_history_per_client)
        )
        self._lock = threading.RLock()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def observe(
        self,
        client_id: str,
        features: NDArray[np.float64],
        confidence: float,
        *,
        timestamp: datetime | None = None,
    ) -> QueryRecord:
        """Record one prediction request and return the stored record."""
        record = QueryRecord(
            client_id=client_id,
            features=np.asarray(features, dtype=np.float64).ravel(),
            confidence=confidence,
            timestamp=timestamp or _utcnow(),
        )
        with self._lock:
            self._history[client_id].append(record)
        return record

    def reset(self, client_id: str | None = None) -> None:
        """Forget history for one client, or for all clients."""
        with self._lock:
            if client_id is None:
                self._history.clear()
            else:
                self._history.pop(client_id, None)

    def clients(self) -> list[str]:
        """Return every client with recorded history."""
        with self._lock:
            return list(self._history)

    # ── Assessment ───────────────────────────────────────────────────────────

    def assess(self, client_id: str, *, now: datetime | None = None) -> ExtractionVerdict:
        """Return the extraction-risk verdict for ``client_id``."""
        moment = now or _utcnow()
        cutoff = moment - self.window
        with self._lock:
            records = [r for r in self._history.get(client_id, ()) if r.timestamp >= cutoff]

        if len(records) < self.min_queries:
            return ExtractionVerdict(
                client_id=client_id,
                risk=ExtractionRisk.LOW,
                score=0.0,
                signals=(),
                query_count=len(records),
                queries_per_minute=0.0,
                mean_confidence=float(np.mean([r.confidence for r in records])) if records else 0.0,
                coverage_uniformity=0.0,
                near_duplicate_ratio=0.0,
                details={"reason": "insufficient history"},
            )

        features = np.vstack([r.features for r in records])
        confidences = np.array([r.confidence for r in records])
        span_minutes = max(
            (records[-1].timestamp - records[0].timestamp).total_seconds() / 60.0, 1e-6
        )
        rate = len(records) / span_minutes

        uniformity = _coverage_uniformity(features)
        probe_ratio = float((confidences < self.low_confidence_threshold).mean())
        duplicate_ratio = _near_duplicate_ratio(features, self.near_duplicate_epsilon)

        signals: list[ExtractionSignal] = []
        score = 0.0
        if rate > self.max_queries_per_minute:
            signals.append(ExtractionSignal.HIGH_VOLUME)
            score += 0.2 + 0.1 * min(rate / self.max_queries_per_minute - 1.0, 1.0)
        if uniformity > self.uniformity_threshold:
            signals.append(ExtractionSignal.UNIFORM_COVERAGE)
            score += 0.2
        if probe_ratio > self.boundary_probe_ratio:
            signals.append(ExtractionSignal.BOUNDARY_PROBING)
            score += 0.2
        if duplicate_ratio > self.near_duplicate_ratio:
            signals.append(ExtractionSignal.NEAR_DUPLICATE_PROBING)
            score += 0.3

        score = min(score, 1.0)
        return ExtractionVerdict(
            client_id=client_id,
            risk=_risk_for(score),
            score=score,
            signals=tuple(signals),
            query_count=len(records),
            queries_per_minute=rate,
            mean_confidence=float(confidences.mean()),
            coverage_uniformity=uniformity,
            near_duplicate_ratio=duplicate_ratio,
            details={
                "boundary_probe_ratio": probe_ratio,
                "window_minutes": self.window.total_seconds() / 60.0,
            },
        )

    def assess_all(self, *, now: datetime | None = None) -> list[ExtractionVerdict]:
        """Return verdicts for every known client, riskiest first."""
        verdicts = [self.assess(client, now=now) for client in self.clients()]
        verdicts.sort(key=lambda verdict: verdict.score, reverse=True)
        return verdicts

    def report(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Return a summary of every client's extraction risk."""
        verdicts = self.assess_all(now=now)
        return {
            "generated_at": (now or _utcnow()).isoformat(),
            "client_count": len(verdicts),
            "at_risk_count": sum(
                1 for v in verdicts if v.risk in (ExtractionRisk.HIGH, ExtractionRisk.CRITICAL)
            ),
            "verdicts": [verdict.to_dict() for verdict in verdicts],
        }


def _coverage_uniformity(features: NDArray[np.float64]) -> float:
    """Return how uniformly queries cover the feature space, in ``[0, 1]``.

    Compares the observed per-feature standard deviation to that of a uniform
    distribution over the observed range.  Synthetic probe grids approach 1;
    real traffic, which clusters, sits well below.
    """
    if features.shape[0] < 2:
        return 0.0
    spans = features.max(axis=0) - features.min(axis=0)
    active = spans > 1e-12
    if not active.any():
        return 0.0
    observed = features[:, active].std(axis=0)
    # std of Uniform(0, span) is span / sqrt(12).
    expected = spans[active] / np.sqrt(12.0)
    return float(np.clip(np.mean(observed / expected), 0.0, 1.0))


def _near_duplicate_ratio(features: NDArray[np.float64], epsilon: float) -> float:
    """Return the share of queries that nearly repeat an earlier query.

    Compares each query with its immediate predecessor in arrival order, which
    is where finite-difference probing puts its paired queries and keeps the
    check linear in the number of observations.
    """
    if features.shape[0] < 2:
        return 0.0
    deltas = np.abs(np.diff(features, axis=0))
    per_pair_max = deltas.max(axis=1)
    changed_features = (deltas > epsilon).sum(axis=1)
    # Identical or single-feature-perturbed neighbours both indicate probing.
    near = (per_pair_max <= epsilon) | (changed_features == 1)
    return float(near.mean())


def _risk_for(score: float) -> ExtractionRisk:
    """Map a risk score onto a risk band."""
    if score >= 0.8:
        return ExtractionRisk.CRITICAL
    if score >= 0.5:
        return ExtractionRisk.HIGH
    if score >= 0.25:
        return ExtractionRisk.MEDIUM
    return ExtractionRisk.LOW
