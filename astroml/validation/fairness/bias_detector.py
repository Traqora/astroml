"""Bias detection and intersectional analysis for model fairness.

Provides the BiasDetector class for detecting bias across protected
attributes, performing intersectional analysis, and generating structured
bias reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from astroml.validation.fairness.metrics import FairnessMetrics

logger = logging.getLogger(__name__)


@dataclass
class PerAttributeResult:
    """Bias detection result for a single protected attribute.

    Attributes:
        attribute_name: Name of the protected attribute.
        metrics: Dictionary of fairness metric results.
        bias_detected: Whether bias was detected for this attribute.
        severity: Severity score (0.0 to 1.0).
    """

    attribute_name: str
    metrics: dict[str, Any]
    bias_detected: bool
    severity: float


@dataclass
class IntersectionalResult:
    """Bias detection result for an intersection of attributes.

    Attributes:
        groups: Tuple of attribute values defining the intersection.
        metrics: Dictionary of fairness metric results.
        bias_detected: Whether bias was detected for this intersection.
        sample_size: Number of samples in this intersection group.
    """

    groups: tuple[str, ...]
    metrics: dict[str, Any]
    bias_detected: bool
    sample_size: int


@dataclass
class BiasDetectionResult:
    """Complete bias detection result.

    Attributes:
        overall_bias_detected: Whether any bias was detected overall.
        per_attribute: List of per-attribute bias results.
        intersectional_results: List of intersectional analysis results.
        severity: Overall severity score (0.0 to 1.0).
        details: Optional additional context.
    """

    overall_bias_detected: bool
    per_attribute: list[PerAttributeResult] = field(default_factory=list)
    intersectional_results: list[IntersectionalResult] = field(default_factory=list)
    severity: float = 0.0
    details: str | None = None


@dataclass
class FeatureBiasResult:
    """Result of feature-level bias analysis.

    Attributes:
        feature_name: Name of the feature analyzed.
        group_means: Mean feature values per group.
        bias_detected: Whether bias was detected.
        max_difference: Maximum difference in means across groups.
    """

    feature_name: str
    group_means: dict[str, float]
    bias_detected: bool
    max_difference: float


def _compute_severity(
    per_attribute: list[PerAttributeResult],
) -> float:
    """Compute overall severity score from per-attribute results.

    Args:
        per_attribute: List of per-attribute bias results.

    Returns:
        Severity score between 0.0 and 1.0.
    """
    if not per_attribute:
        return 0.0
    return max(r.severity for r in per_attribute)


def _compute_attribute_severity(
    metrics: dict[str, Any],
) -> float:
    """Compute severity score for an attribute based on its metrics.

    Args:
        metrics: Dictionary of fairness metric results with 'passed' keys.

    Returns:
        Severity score between 0.0 and 1.0.
    """
    if not metrics:
        return 0.0
    failed = sum(1 for m in metrics.values() if hasattr(m, "passed") and not m.passed)
    return failed / len(metrics)


class BiasDetector:
    """Detect bias in model predictions across protected attributes.

    Wraps FairnessMetrics to provide high-level bias detection, intersectional
    analysis, and structured reporting capabilities.
    """

    def __init__(self) -> None:
        self._metrics = FairnessMetrics()

    def detect_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        attributes: list[str] | None = None,
    ) -> BiasDetectionResult:
        """Detect bias across protected attributes.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Array of protected attribute values. For
                multiple attributes, use a 2D array where each column is an
                attribute.
            attributes: Names of the protected attributes. If None, numeric
                indices are used.

        Returns:
            BiasDetectionResult with per-attribute and overall bias info.
        """
        sensitive_features = np.asarray(sensitive_features)
        if sensitive_features.ndim == 1:
            sensitive_features = sensitive_features.reshape(-1, 1)
            if attributes is not None and len(attributes) == 1:
                pass
            attributes = attributes or ["attribute_0"]

        n_attrs = sensitive_features.shape[1]
        if attributes is None:
            attributes = [f"attribute_{i}" for i in range(n_attrs)]
        elif len(attributes) != n_attrs:
            raise ValueError(
                f"Number of attribute names ({len(attributes)}) does not match "
                f"number of columns ({n_attrs})"
            )

        per_attribute: list[PerAttributeResult] = []
        for i, attr_name in enumerate(attributes):
            attr_values = sensitive_features[:, i]
            all_metrics = self._metrics.compute_all(y_true, y_pred, attr_values)
            bias_detected = any(not m.passed for m in all_metrics.values())
            severity = _compute_attribute_severity(all_metrics)
            per_attribute.append(
                PerAttributeResult(
                    attribute_name=attr_name,
                    metrics={
                        name: {
                            "value": m.value,
                            "threshold": m.threshold,
                            "passed": m.passed,
                            "group_metrics": m.group_metrics,
                        }
                        for name, m in all_metrics.items()
                    },
                    bias_detected=bias_detected,
                    severity=severity,
                )
            )

        overall_bias_detected = any(r.bias_detected for r in per_attribute)
        severity = _compute_severity(per_attribute)

        return BiasDetectionResult(
            overall_bias_detected=overall_bias_detected,
            per_attribute=per_attribute,
            severity=severity,
        )

    def intersectional_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        intersection_groups: list[list[int]] | None = None,
    ) -> list[IntersectionalResult]:
        """Perform intersectional bias analysis across combinations of attributes.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: 2D array where each column is a protected
                attribute.
            intersection_groups: List of lists of attribute column indices to
                combine. If None, all pairwise combinations are analyzed.

        Returns:
            List of IntersectionalResult for each intersection group.
        """
        sensitive_features = np.asarray(sensitive_features)
        if sensitive_features.ndim == 1:
            sensitive_features = sensitive_features.reshape(-1, 1)

        n_attrs = sensitive_features.shape[1]
        if intersection_groups is None:
            intersection_groups = [list(pair) for pair in combinations(range(n_attrs), 2)]

        results: list[IntersectionalResult] = []
        for group_indices in intersection_groups:
            if len(group_indices) < 2:
                continue
            combined_values = _combine_attributes(sensitive_features, group_indices)
            unique_combos = np.unique(combined_values)

            for combo in unique_combos:
                mask = combined_values == combo
                if mask.sum() < 2:
                    continue

                subset_y_true = y_true[mask]
                subset_y_pred = y_pred[mask]

                all_metrics = self._metrics.compute_all(
                    subset_y_true, subset_y_pred, np.ones(mask.sum())
                )

                bias_detected = any(not m.passed for m in all_metrics.values())

                group_values = tuple(str(sensitive_features[mask][0, idx]) for idx in group_indices)

                results.append(
                    IntersectionalResult(
                        groups=group_values,
                        metrics={
                            name: {
                                "value": m.value,
                                "threshold": m.threshold,
                                "passed": m.passed,
                            }
                            for name, m in all_metrics.items()
                        },
                        bias_detected=bias_detected,
                        sample_size=int(mask.sum()),
                    )
                )

        return results

    def feature_bias_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> list[FeatureBiasResult]:
        """Analyze bias in feature distributions across groups.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (unused, kept for API consistency).
            sensitive_features: Protected attribute values.

        Returns:
            List of FeatureBiasResult for each feature.
        """
        sensitive_features = np.asarray(sensitive_features).ravel()
        groups = np.unique(sensitive_features)
        n_features = X.shape[1]
        results: list[FeatureBiasResult] = []

        for f_idx in range(n_features):
            feature = X[:, f_idx]
            group_means: dict[str, float] = {}
            for group in groups:
                mask = sensitive_features == group
                if mask.sum() > 0:
                    group_means[str(group)] = float(np.mean(feature[mask]))
                else:
                    group_means[str(group)] = 0.0

            means = list(group_means.values())
            max_diff = max(means) - min(means) if means else 0.0
            bias_detected = (
                max_diff > 0.5 * max(abs(m) for m in means)
                if means and max(abs(m) for m in means) > 0
                else False
            )

            results.append(
                FeatureBiasResult(
                    feature_name=f"feature_{f_idx}",
                    group_means=group_means,
                    bias_detected=bias_detected,
                    max_difference=max_diff,
                )
            )

        return results

    def distribution_alignment(
        self,
        X: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> dict[str, Any]:
        """Check distribution alignment of features across groups.

        Args:
            X: Feature matrix (n_samples, n_features).
            sensitive_features: Protected attribute values.

        Returns:
            Dictionary with alignment results per feature and overall score.
        """
        sensitive_features = np.asarray(sensitive_features).ravel()
        groups = np.unique(sensitive_features)
        n_features = X.shape[1]

        feature_stats: dict[str, dict[str, dict[str, float]]] = {}
        alignment_scores: list[float] = []

        for f_idx in range(n_features):
            feature = X[:, f_idx]
            fname = f"feature_{f_idx}"
            feature_stats[fname] = {}
            group_means: list[float] = []

            for group in groups:
                mask = sensitive_features == group
                if mask.sum() > 0:
                    mean = float(np.mean(feature[mask]))
                    std = float(np.std(feature[mask]))
                    feature_stats[fname][str(group)] = {
                        "mean": mean,
                        "std": std,
                        "count": int(mask.sum()),
                    }
                    group_means.append(mean)

            if len(group_means) > 1:
                max_mean = max(group_means)
                min_mean = min(group_means)
                overall_mean = float(np.mean(feature))
                if overall_mean != 0:
                    alignment_scores.append(
                        1.0 - min(1.0, (max_mean - min_mean) / abs(overall_mean))
                    )
                else:
                    alignment_scores.append(1.0)

        overall_alignment = float(np.mean(alignment_scores)) if alignment_scores else 1.0

        return {
            "feature_stats": feature_stats,
            "overall_alignment_score": overall_alignment,
            "n_groups": len(groups),
        }

    def report_bias(
        self,
        bias_results: BiasDetectionResult,
    ) -> dict[str, Any]:
        """Generate a structured bias report.

        Args:
            bias_results: BiasDetectionResult to report on.

        Returns:
            Dictionary containing the structured report.
        """
        report: dict[str, Any] = {
            "overall_bias_detected": bias_results.overall_bias_detected,
            "severity": bias_results.severity,
            "per_attribute_breakdown": [],
            "intersectional_findings": [
                {
                    "groups": list(r.groups),
                    "bias_detected": r.bias_detected,
                    "sample_size": r.sample_size,
                }
                for r in bias_results.intersectional_results
            ],
            "recommendations": self._generate_recommendations(bias_results),
        }

        for attr in bias_results.per_attribute:
            attr_report: dict[str, Any] = {
                "attribute": attr.attribute_name,
                "bias_detected": attr.bias_detected,
                "severity": attr.severity,
                "metrics": {},
            }
            for metric_name, metric_data in attr.metrics.items():
                attr_report["metrics"][metric_name] = {
                    "value": metric_data.get("value"),
                    "passed": metric_data.get("passed"),
                }
            report["per_attribute_breakdown"].append(attr_report)

        return report

    def _generate_recommendations(
        self,
        bias_results: BiasDetectionResult,
    ) -> list[str]:
        """Generate mitigation recommendations based on bias results.

        Args:
            bias_results: BiasDetectionResult to generate recommendations for.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []
        if not bias_results.overall_bias_detected:
            recommendations.append("No significant bias detected; no mitigation required.")
            return recommendations

        recommendations.append("Bias detected. Consider using bias mitigation techniques.")
        for attr in bias_results.per_attribute:
            if attr.bias_detected:
                recommendations.append(
                    f"Address bias in attribute '{attr.attribute_name}' "
                    f"(severity: {attr.severity:.2f})."
                )

        recommendations.append(
            "Recommended strategies: reweighing, adversarial debiasing, "
            "or equalized odds post-processing."
        )
        return recommendations


def _combine_attributes(
    sensitive_features: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    """Combine multiple attribute columns into a single compound key.

    Args:
        sensitive_features: 2D array of attribute values.
        indices: Column indices to combine.

    Returns:
        1D array of combined string keys.
    """
    parts = []
    for idx in indices:
        parts.append(sensitive_features[:, idx].astype(str))
    result = parts[0]
    for p in parts[1:]:
        result = np.char.add(np.char.add(result, "_"), p)
    return result
