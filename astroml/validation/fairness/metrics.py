"""Fairness metrics for binary classification model evaluation.

Provides quantitative fairness metrics that measure bias across protected
attribute groups, including demographic parity, equal opportunity, equalized
odds, and disparate impact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default fairness thresholds
DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
EQUAL_OPPORTUNITY_THRESHOLD = 0.1
EQUALIZED_ODDS_THRESHOLD = 0.1
DISPARATE_IMPACT_LOWER = 0.8
DISPARATE_IMPACT_UPPER = 1.25


@dataclass
class FairnessMetricResult:
    """Result of a single fairness metric computation.

    Attributes:
        metric_name: Name of the fairness metric.
        value: Overall metric value.
        threshold: Threshold used for the metric.
        passed: Whether the metric passed the threshold check.
        group_metrics: Mapping of group label to metric value for that group.
        details: Optional additional information.
    """

    metric_name: str
    value: float
    threshold: float | tuple[float, float] | None
    passed: bool
    group_metrics: dict[str, float] = field(default_factory=dict)
    details: str | None = None


def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> None:
    """Validate input arrays have consistent shapes and contain binary labels.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive_features: Protected attribute values.

    Raises:
        ValueError: If inputs have inconsistent shapes or invalid values.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}")
    if y_true.shape != sensitive_features.shape:
        raise ValueError(
            f"y_true shape {y_true.shape} does not match sensitive_features "
            f"shape {sensitive_features.shape}"
        )
    if len(y_true) == 0:
        raise ValueError("Input arrays are empty")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must contain only binary labels (0/1)")
    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("y_pred must contain only binary labels (0/1)")


def _get_group_mask(
    sensitive_features: np.ndarray,
    group: Any,
) -> np.ndarray:
    """Return boolean mask for samples belonging to a given group.

    Args:
        sensitive_features: Array of protected attribute values.
        group: Group label to mask.

    Returns:
        Boolean array indicating membership in the group.
    """
    return sensitive_features == group


def _positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute the positive prediction rate (proportion of predicted positives).

    Args:
        y_true: Ground truth labels (unused, kept for API consistency).
        y_pred: Predicted labels.
        mask: Optional boolean mask to subset the data.

    Returns:
        Positive prediction rate.
    """
    if mask is not None:
        y_pred = y_pred[mask]
    if len(y_pred) == 0:
        return 0.0
    return float(np.mean(y_pred))


def _true_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute the true positive rate (recall) for the positive class.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        mask: Optional boolean mask to subset the data.

    Returns:
        True positive rate.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    positives = y_true == 1
    if positives.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[positives] == 1))


def _false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute the false positive rate.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        mask: Optional boolean mask to subset the data.

    Returns:
        False positive rate.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[negatives] == 1))


class FairnessMetrics:
    """Compute standard fairness metrics for binary classification.

    Supports evaluation of model fairness across protected attribute groups
    using metrics such as demographic parity, equal opportunity, equalized
    odds, and disparate impact.
    """

    def demographic_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> FairnessMetricResult:
        """Compute demographic parity difference and ratio.

        Demographic parity is satisfied if the prediction rate is equal across
        groups. The difference is the max-min prediction rate gap.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Protected attribute values.

        Returns:
            FairnessMetricResult with demographic parity metrics.
        """
        _validate_inputs(y_true, y_pred, sensitive_features)
        groups = np.unique(sensitive_features)
        group_metrics: dict[str, float] = {}
        for group in groups:
            mask = _get_group_mask(sensitive_features, group)
            group_metrics[str(group)] = _positive_rate(y_true, y_pred, mask)

        rates = list(group_metrics.values())
        value = max(rates) - min(rates)
        passed = value < DEMOGRAPHIC_PARITY_THRESHOLD

        return FairnessMetricResult(
            metric_name="demographic_parity",
            value=value,
            threshold=DEMOGRAPHIC_PARITY_THRESHOLD,
            passed=passed,
            group_metrics=group_metrics,
            details=(
                f"Demographic parity difference={value:.4f} "
                f"(threshold={DEMOGRAPHIC_PARITY_THRESHOLD})"
            ),
        )

    def equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        positive_class: int = 1,
    ) -> FairnessMetricResult:
        """Compute equal opportunity difference.

        Equal opportunity requires equal true positive rates across groups.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Protected attribute values.
            positive_class: The positive class label (default 1).

        Returns:
            FairnessMetricResult with equal opportunity metrics.
        """
        _validate_inputs(y_true, y_pred, sensitive_features)
        groups = np.unique(sensitive_features)
        group_metrics: dict[str, float] = {}
        for group in groups:
            mask = _get_group_mask(sensitive_features, group)
            group_metrics[str(group)] = _true_positive_rate(y_true, y_pred, mask)

        rates = list(group_metrics.values())
        value = max(rates) - min(rates)
        passed = value < EQUAL_OPPORTUNITY_THRESHOLD

        return FairnessMetricResult(
            metric_name="equal_opportunity",
            value=value,
            threshold=EQUAL_OPPORTUNITY_THRESHOLD,
            passed=passed,
            group_metrics=group_metrics,
            details=(
                f"Equal opportunity difference={value:.4f} "
                f"(threshold={EQUAL_OPPORTUNITY_THRESHOLD})"
            ),
        )

    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> FairnessMetricResult:
        """Compute equalized odds difference.

        Equalized odds requires equal TPR and FPR across groups. The reported
        value is the maximum of the TPR difference and FPR difference.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Protected attribute values.

        Returns:
            FairnessMetricResult with equalized odds metrics.
        """
        _validate_inputs(y_true, y_pred, sensitive_features)
        groups = np.unique(sensitive_features)
        tpr_metrics: dict[str, float] = {}
        fpr_metrics: dict[str, float] = {}
        for group in groups:
            mask = _get_group_mask(sensitive_features, group)
            tpr_metrics[str(group)] = _true_positive_rate(y_true, y_pred, mask)
            fpr_metrics[str(group)] = _false_positive_rate(y_true, y_pred, mask)

        tpr_rates = list(tpr_metrics.values())
        fpr_rates = list(fpr_metrics.values())
        tpr_diff = max(tpr_rates) - min(tpr_rates)
        fpr_diff = max(fpr_rates) - min(fpr_rates)
        value = max(tpr_diff, fpr_diff)
        passed = value < EQUALIZED_ODDS_THRESHOLD

        group_metrics: dict[str, float] = {}
        for g in groups:
            gs = str(g)
            group_metrics[gs] = max(tpr_metrics[gs], fpr_metrics[gs])

        return FairnessMetricResult(
            metric_name="equalized_odds",
            value=value,
            threshold=EQUALIZED_ODDS_THRESHOLD,
            passed=passed,
            group_metrics=group_metrics,
            details=(
                f"Equalized odds difference={value:.4f} "
                f"(TPR diff={tpr_diff:.4f}, FPR diff={fpr_diff:.4f}, "
                f"threshold={EQUALIZED_ODDS_THRESHOLD})"
            ),
        )

    def disparate_impact(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> FairnessMetricResult:
        """Compute disparate impact ratio.

        Disparate impact measures the ratio of the positive prediction rate
        for the most privileged group to the least privileged group. A ratio
        below 0.8 or above 1.25 indicates potential disparate impact.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Protected attribute values.

        Returns:
            FairnessMetricResult with disparate impact metrics.
        """
        _validate_inputs(y_true, y_pred, sensitive_features)
        groups = np.unique(sensitive_features)
        group_metrics: dict[str, float] = {}
        for group in groups:
            mask = _get_group_mask(sensitive_features, group)
            group_metrics[str(group)] = _positive_rate(y_true, y_pred, mask)

        rates = list(group_metrics.values())
        if min(rates) == 0:
            value = float("inf")
        else:
            value = max(rates) / min(rates)

        threshold = (DISPARATE_IMPACT_LOWER, DISPARATE_IMPACT_UPPER)
        passed = DISPARATE_IMPACT_LOWER <= value <= DISPARATE_IMPACT_UPPER

        return FairnessMetricResult(
            metric_name="disparate_impact",
            value=value,
            threshold=threshold,
            passed=passed,
            group_metrics=group_metrics,
            details=(
                f"Disparate impact ratio={value:.4f} "
                f"(acceptable range: {DISPARATE_IMPACT_LOWER}-"
                f"{DISPARATE_IMPACT_UPPER})"
            ),
        )

    def statistical_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> FairnessMetricResult:
        """Alias for demographic parity.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Protected attribute values.

        Returns:
            FairnessMetricResult with demographic parity metrics.
        """
        return self.demographic_parity(y_true, y_pred, sensitive_features)

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> dict[str, FairnessMetricResult]:
        """Compute all fairness metrics at once.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            sensitive_features: Protected attribute values.

        Returns:
            Dictionary mapping metric names to FairnessMetricResult objects.
        """
        return {
            "demographic_parity": self.demographic_parity(y_true, y_pred, sensitive_features),
            "equal_opportunity": self.equal_opportunity(y_true, y_pred, sensitive_features),
            "equalized_odds": self.equalized_odds(y_true, y_pred, sensitive_features),
            "disparate_impact": self.disparate_impact(y_true, y_pred, sensitive_features),
        }
