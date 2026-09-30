"""Resource utilization analysis for ML workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ResourceMetrics:
    """Metrics for a specific resource type."""

    cpu_utilization_percent: float
    memory_utilization_percent: float
    gpu_utilization_percent: float | None = None
    memory_bytes_used: int = 0
    memory_bytes_total: int = 0


@dataclass
class WorkloadResourceProfile:
    """Resource profile for a specific ML workload."""

    workload_id: str
    workload_type: str  # e.g., 'training', 'inference', 'batch'
    metrics: ResourceMetrics
    instance_type: str
    duration_seconds: float


class ResourceAnalyzer:
    """Analyzes resource utilization of ML workloads."""

    def __init__(self) -> None:
        pass

    def analyze_utilization(self, profile: WorkloadResourceProfile) -> Dict[str, float]:
        """
        Analyze resource utilization and return utilization scores.
        Scores are 0.0 (idle) to 1.0 (fully utilized).
        """
        scores = {
            "cpu_score": profile.metrics.cpu_utilization_percent / 100.0,
            "memory_score": profile.metrics.memory_utilization_percent / 100.0,
        }
        if profile.metrics.gpu_utilization_percent is not None:
            scores["gpu_score"] = profile.metrics.gpu_utilization_percent / 100.0

        return scores

    def identify_underutilized_resources(
        self, profiles: List[WorkloadResourceProfile], threshold: float = 0.3
    ) -> List[WorkloadResourceProfile]:
        """
        Identify workloads that are consistently underutilizing their allocated resources.
        A workload is considered underutilized if all active resource scores are below the threshold.
        """
        underutilized = []
        for profile in profiles:
            scores = self.analyze_utilization(profile)
            if all(score < threshold for score in scores.values()):
                underutilized.append(profile)
        return underutilized
