"""Metrics comparison tooling across model versions.

Provides a CLI/report comparing candidate versions across metrics
(latency, F1, drift) to guide promotion decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VersionMetrics:
    """Metrics for a single model version."""

    version_id: str
    model_name: str
    latency_ms: float
    f1_score: float
    drift_score: float
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "f1_score": self.f1_score,
            "drift_score": self.drift_score,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            **self.additional_metrics,
        }


class MetricsComparator:
    """Compares candidate versions across metrics to guide promotion decisions."""

    def __init__(self) -> None:
        self._versions: dict[str, VersionMetrics] = {}

    def add_version(self, metrics: VersionMetrics) -> None:
        """Add a version's metrics for comparison."""
        self._versions[metrics.version_id] = metrics

    def remove_version(self, version_id: str) -> None:
        """Remove a version from comparison."""
        self._versions.pop(version_id, None)

    def compare(self) -> ComparisonReport:
        """Generate a comparison report across all registered versions.

        Ranks versions by F1 score (primary), then by accuracy,
        then by latency (lower is better). Includes drift warnings
        and promotion recommendations.
        """
        versions = list(self._versions.values())
        if not versions:
            return ComparisonReport(version_ids=[], best=None)

        # Sort by F1 desc, accuracy desc, latency asc
        sorted_versions = sorted(
            versions,
            key=lambda v: (-v.f1_score, -v.accuracy, v.latency_ms),
        )

        best = sorted_versions[0]
        recommendations = self._generate_recommendations(sorted_versions, best)

        return ComparisonReport(
            version_ids=[v.version_id for v in sorted_versions],
            best=best.version_id,
            recommendations=recommendations,
            all_metrics={v.version_id: v.to_dict() for v in sorted_versions},
        )

    def _generate_recommendations(
        self, sorted_versions: list[VersionMetrics], best: VersionMetrics
    ) -> list[str]:
        """Generate promotion recommendations based on metrics."""
        recommendations = []

        if best.drift_score > 0.5:
            recommendations.append(
                f"Version {best.version_id} has high drift ({best.drift_score:.3f}); "
                "consider retraining before promotion"
            )

        if best.f1_score < 0.7:
            recommendations.append(
                f"Version {best.version_id} has low F1 ({best.f1_score:.3f}); "
                "may not meet promotion threshold"
            )

        for v in sorted_versions[1:]:
            if v.f1_score > best.f1_score * 0.9 and v.latency_ms < best.latency_ms:
                recommendations.append(
                    f"Version {v.version_id} is a competitive alternative: "
                    f"F1={v.f1_score:.3f}, latency={v.latency_ms}ms"
                )

        if not recommendations:
            recommendations.append(
                f"Version {best.version_id} is the leading candidate for promotion"
            )

        return recommendations


@dataclass
class ComparisonReport:
    """Result of comparing multiple model versions."""

    version_ids: list[str]
    best: str | None
    recommendations: list[str]
    all_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_ids": self.version_ids,
            "best": self.best,
            "recommendations": self.recommendations,
            "all_metrics": self.all_metrics,
        }
