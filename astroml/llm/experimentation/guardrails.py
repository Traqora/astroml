"""Safety guardrails for experiments to prevent negative impacts."""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class SafetyMetrics:
    """Safety metrics for an experiment variant."""
    hallucination_rate: float
    toxicity_score: float
    anomaly_count: int
    latency_p99_ms: float


class SafetyGuardrails:
    """
    Monitor and enforce safety constraints during experiments.
    
    Prevents deploying models with safety regressions.
    """

    def __init__(self):
        """Initialize safety guardrails."""
        self.thresholds = {
            "max_hallucination_rate": 0.05,  # 5%
            "max_toxicity_score": 0.10,
            "max_anomalies_per_100": 2,
            "max_latency_increase": 0.30,  # 30% slower acceptable
        }

    def check_variant_safety(
        self,
        variant_name: str,
        safety_metrics: SafetyMetrics,
        baseline_metrics: SafetyMetrics = None,
    ) -> Dict[str, Any]:
        """
        Check if variant meets safety constraints.

        Args:
            variant_name: Name of variant
            safety_metrics: Current safety metrics
            baseline_metrics: Baseline metrics to compare against

        Returns:
            Safety check results
        """
        issues = []

        # Check hallucination rate
        if safety_metrics.hallucination_rate > self.thresholds["max_hallucination_rate"]:
            issues.append(f"Hallucination rate too high: {safety_metrics.hallucination_rate:.1%}")

        # Check toxicity
        if safety_metrics.toxicity_score > self.thresholds["max_toxicity_score"]:
            issues.append(f"Toxicity score too high: {safety_metrics.toxicity_score:.3f}")

        # Check latency regression if baseline provided
        if baseline_metrics:
            latency_increase = (
                (safety_metrics.latency_p99_ms - baseline_metrics.latency_p99_ms) 
                / baseline_metrics.latency_p99_ms
            )
            if latency_increase > self.thresholds["max_latency_increase"]:
                issues.append(f"Latency increased {latency_increase:.1%}")

        return {
            "variant": variant_name,
            "is_safe": len(issues) == 0,
            "issues": issues,
            "metrics": {
                "hallucination_rate": safety_metrics.hallucination_rate,
                "toxicity_score": safety_metrics.toxicity_score,
                "anomalies": safety_metrics.anomaly_count,
                "latency_p99_ms": safety_metrics.latency_p99_ms,
            },
        }

    def should_rollback(
        self,
        variant_name: str,
        safety_check: Dict[str, Any],
    ) -> bool:
        """
        Determine if experiment should rollback.

        Args:
            variant_name: Variant name
            safety_check: Safety check results

        Returns:
            True if rollback is recommended
        """
        return not safety_check["is_safe"]

    def set_threshold(self, metric_name: str, value: float) -> None:
        """
        Set safety threshold for a metric.

        Args:
            metric_name: Name of metric
            value: Threshold value
        """
        if f"max_{metric_name}" in self.thresholds:
            self.thresholds[f"max_{metric_name}"] = value

    def get_thresholds(self) -> Dict[str, float]:
        """Get all safety thresholds."""
        return self.thresholds.copy()

    def monitor_continuous(
        self,
        observations: List[Dict[str, Any]],
        window_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Continuously monitor safety during experiment.

        Args:
            observations: Recent observations
            window_size: Number of recent observations to analyze

        Returns:
            Continuous monitoring results
        """
        recent = observations[-window_size:]

        # Simulate monitoring
        anomalies = sum(1 for obs in recent if obs.get("anomaly", False))

        return {
            "observations_analyzed": len(recent),
            "anomalies_detected": anomalies,
            "anomaly_rate": anomalies / len(recent) if recent else 0,
            "alert_level": "low" if anomalies < 2 else "high",
        }
