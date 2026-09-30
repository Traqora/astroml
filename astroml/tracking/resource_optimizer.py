"""Resource utilization analysis and cost-optimization recommendations.

Resolves part of #647.

Callers feed :class:`ResourceUsageSample` observations (from Prometheus, the
Kubernetes metrics API, ``nvidia-smi``, or anywhere else) into
:class:`ResourceOptimizer`, which detects under-utilised, over-utilised and
idle resources and returns ranked, actionable :class:`Recommendation` objects
with an estimated monthly saving.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from statistics import fmean
from typing import Any

from astroml.tracking.cost_tracker import (
    DEFAULT_COMPUTE_PRICES_USD_PER_HOUR,
    CostTracker,
    ResourceType,
    utcnow,
)

__all__ = [
    "OptimizationKind",
    "Recommendation",
    "ResourceOptimizer",
    "ResourceUsageSample",
    "Severity",
]

#: Below this mean utilization a resource is considered over-provisioned.
UNDERUTILIZED_THRESHOLD = 0.35
#: Above this mean utilization a resource is considered saturated.
SATURATED_THRESHOLD = 0.90
#: Below this mean utilization a resource is considered effectively idle.
IDLE_THRESHOLD = 0.05
#: Minimum samples before a resource is judged at all.
MIN_SAMPLES = 3
#: Hours in an average month, used to annualise hourly prices.
HOURS_PER_MONTH = 730.0


class OptimizationKind(str, Enum):
    """Category of a recommendation."""

    RIGHTSIZE = "rightsize"
    SCALE_UP = "scale_up"
    SHUTDOWN_IDLE = "shutdown_idle"
    SPOT_INSTANCE = "spot_instance"
    BATCH_CONSOLIDATION = "batch_consolidation"


class Severity(str, Enum):
    """How urgently a recommendation should be acted on."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ResourceUsageSample:
    """A single utilization observation for one resource instance.

    Utilization values are fractions in ``[0, 1]``.
    """

    resource_id: str
    resource_type: ResourceType
    utilization: float
    memory_utilization: float | None = None
    timestamp: datetime = field(default_factory=utcnow)
    project: str | None = None
    team: str | None = None
    interruptible: bool = False

    def __post_init__(self) -> None:
        """Validate utilization bounds."""
        if not 0.0 <= self.utilization <= 1.0:
            raise ValueError("utilization must be within [0, 1]")
        if self.memory_utilization is not None and not 0.0 <= self.memory_utilization <= 1.0:
            raise ValueError("memory_utilization must be within [0, 1]")


@dataclass(frozen=True)
class Recommendation:
    """An actionable, costed optimization suggestion."""

    resource_id: str
    resource_type: ResourceType
    kind: OptimizationKind
    severity: Severity
    message: str
    mean_utilization: float
    peak_utilization: float
    sample_count: int
    estimated_monthly_saving_usd: float
    project: str | None = None
    team: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the recommendation."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "message": self.message,
            "mean_utilization": self.mean_utilization,
            "peak_utilization": self.peak_utilization,
            "sample_count": self.sample_count,
            "estimated_monthly_saving_usd": self.estimated_monthly_saving_usd,
            "project": self.project,
            "team": self.team,
        }


class ResourceOptimizer:
    """Turns utilization samples into ranked cost-saving recommendations.

    Parameters
    ----------
    tracker:
        Optional :class:`CostTracker` used to source hourly prices so that the
        optimizer honours deployment-specific pricing overrides.
    spot_discount:
        Assumed saving when moving an interruptible workload to spot capacity.
    """

    def __init__(
        self,
        tracker: CostTracker | None = None,
        *,
        spot_discount: float = 0.7,
        max_samples_per_resource: int = 10_000,
    ) -> None:
        if not 0.0 < spot_discount < 1.0:
            raise ValueError("spot_discount must be within (0, 1)")
        if max_samples_per_resource <= 0:
            raise ValueError("max_samples_per_resource must be positive")
        self._tracker = tracker
        self._spot_discount = spot_discount
        self._max_samples = max_samples_per_resource
        self._samples: dict[str, list[ResourceUsageSample]] = {}

    # ── Ingestion ────────────────────────────────────────────────────────────

    def record(self, sample: ResourceUsageSample) -> None:
        """Store one utilization sample."""
        bucket = self._samples.setdefault(sample.resource_id, [])
        bucket.append(sample)
        if len(bucket) > self._max_samples:
            del bucket[0]

    def record_many(self, samples: Iterable[ResourceUsageSample]) -> None:
        """Store many utilization samples."""
        for sample in samples:
            self.record(sample)

    def reset(self) -> None:
        """Discard all stored samples."""
        self._samples.clear()

    # ── Analysis ─────────────────────────────────────────────────────────────

    def utilization_summary(self) -> dict[str, dict[str, float]]:
        """Return mean/peak/min utilization and sample count per resource."""
        summary: dict[str, dict[str, float]] = {}
        for resource_id, samples in self._samples.items():
            values = [s.utilization for s in samples]
            summary[resource_id] = {
                "mean": fmean(values),
                "peak": max(values),
                "min": min(values),
                "samples": float(len(values)),
            }
        return summary

    def recommend(self) -> list[Recommendation]:
        """Return recommendations for every analysable resource, costliest first."""
        recommendations: list[Recommendation] = []
        for resource_id, samples in self._samples.items():
            recommendation = self._analyse(resource_id, samples)
            if recommendation is not None:
                recommendations.append(recommendation)
        recommendations.sort(key=lambda r: r.estimated_monthly_saving_usd, reverse=True)
        return recommendations

    def report(self) -> dict[str, Any]:
        """Return a summary of recommendations and total addressable saving."""
        recommendations = self.recommend()
        return {
            "generated_at": utcnow().isoformat(),
            "resource_count": len(self._samples),
            "recommendation_count": len(recommendations),
            "estimated_monthly_saving_usd": sum(
                r.estimated_monthly_saving_usd for r in recommendations
            ),
            "recommendations": [r.to_dict() for r in recommendations],
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _hourly_price(self, resource_type: ResourceType) -> float:
        """Return the USD/hour price for ``resource_type``."""
        if self._tracker is not None:
            return self._tracker.compute_price(resource_type)
        return DEFAULT_COMPUTE_PRICES_USD_PER_HOUR.get(resource_type, 0.0)

    def _analyse(
        self, resource_id: str, samples: Sequence[ResourceUsageSample]
    ) -> Recommendation | None:
        """Return the single most valuable recommendation for one resource."""
        if len(samples) < MIN_SAMPLES:
            return None

        values = [s.utilization for s in samples]
        mean_util = fmean(values)
        peak_util = max(values)
        latest = samples[-1]
        monthly_cost = self._hourly_price(latest.resource_type) * HOURS_PER_MONTH

        if mean_util <= IDLE_THRESHOLD:
            return Recommendation(
                resource_id=resource_id,
                resource_type=latest.resource_type,
                kind=OptimizationKind.SHUTDOWN_IDLE,
                severity=Severity.HIGH,
                message=(
                    f"{resource_id} is idle ({mean_util:.1%} mean utilization over "
                    f"{len(values)} samples); decommission it or move the workload "
                    "to an on-demand job."
                ),
                mean_utilization=mean_util,
                peak_utilization=peak_util,
                sample_count=len(values),
                estimated_monthly_saving_usd=monthly_cost,
                project=latest.project,
                team=latest.team,
            )

        if mean_util < UNDERUTILIZED_THRESHOLD:
            # Headroom we can reclaim, leaving 2x the observed peak as buffer.
            reclaimable = max(0.0, 1.0 - min(1.0, peak_util * 2.0))
            return Recommendation(
                resource_id=resource_id,
                resource_type=latest.resource_type,
                kind=OptimizationKind.RIGHTSIZE,
                severity=Severity.MEDIUM,
                message=(
                    f"{resource_id} averages {mean_util:.1%} utilization with a "
                    f"{peak_util:.1%} peak; move to a smaller instance class."
                ),
                mean_utilization=mean_util,
                peak_utilization=peak_util,
                sample_count=len(values),
                estimated_monthly_saving_usd=monthly_cost * reclaimable,
                project=latest.project,
                team=latest.team,
            )

        if mean_util >= SATURATED_THRESHOLD:
            return Recommendation(
                resource_id=resource_id,
                resource_type=latest.resource_type,
                kind=OptimizationKind.SCALE_UP,
                severity=Severity.HIGH,
                message=(
                    f"{resource_id} is saturated ({mean_util:.1%} mean utilization); "
                    "scale up or shard the workload to avoid queueing delays."
                ),
                mean_utilization=mean_util,
                peak_utilization=peak_util,
                sample_count=len(values),
                estimated_monthly_saving_usd=0.0,
                project=latest.project,
                team=latest.team,
            )

        if latest.interruptible:
            return Recommendation(
                resource_id=resource_id,
                resource_type=latest.resource_type,
                kind=OptimizationKind.SPOT_INSTANCE,
                severity=Severity.LOW,
                message=(
                    f"{resource_id} runs an interruptible workload at "
                    f"{mean_util:.1%} utilization; spot capacity would cut its cost."
                ),
                mean_utilization=mean_util,
                peak_utilization=peak_util,
                sample_count=len(values),
                estimated_monthly_saving_usd=monthly_cost * self._spot_discount,
                project=latest.project,
                team=latest.team,
            )

        return None
