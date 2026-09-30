"""Error analysis for trained classification models.

Provides high-level error metrics (error rate, accuracy, per-class error
rates) plus the indices of misclassified samples so they can be inspected
or fed into downstream debugging tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassErrorMetrics:
    """Per-class error statistics.

    Attributes:
        class_label: The class label this metrics object refers to.
        total: Number of samples with this true label.
        correct: Number of correctly classified samples.
        incorrect: Number of misclassified samples.
        error_rate: Fraction of samples with this label that were misclassified.
    """

    class_label: int
    total: int
    correct: int
    incorrect: int
    error_rate: float


@dataclass
class ErrorAnalysisResult:
    """Result of a full error analysis run.

    Attributes:
        total_samples: Number of evaluated samples.
        error_count: Number of misclassified samples.
        error_rate: Overall misclassification rate.
        accuracy: Overall accuracy.
        error_indices: Indices of misclassified samples.
        class_metrics: Per-class error metrics keyed by class label.
        error_distribution: Mapping of ``(true_label, predicted_label)`` to count.
    """

    total_samples: int
    error_count: int
    error_rate: float
    accuracy: float
    error_indices: np.ndarray
    class_metrics: dict[int, ClassErrorMetrics] = field(default_factory=dict)
    error_distribution: dict[tuple[int, int], int] = field(default_factory=dict)


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate that prediction arrays are aligned and non-empty.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Raises:
        ValueError: If the arrays differ in length or are empty.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")


class ErrorAnalyzer:
    """Analyze classification errors for a model's predictions."""

    def error_indices(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Return the indices of misclassified samples.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Array of indices where predictions differ from ground truth.
        """
        _validate(y_true, y_pred)
        return np.flatnonzero(y_true != y_pred)

    def error_rate_by_class(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[int, ClassErrorMetrics]:
        """Compute error statistics for every class present in ``y_true``.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Mapping of class label to :class:`ClassErrorMetrics`.
        """
        _validate(y_true, y_pred)
        metrics: dict[int, ClassErrorMetrics] = {}
        for label in np.unique(y_true):
            mask = y_true == label
            total = int(np.count_nonzero(mask))
            incorrect = int(np.count_nonzero(y_pred[mask] != label))
            metrics[int(label)] = ClassErrorMetrics(
                class_label=int(label),
                total=total,
                correct=total - incorrect,
                incorrect=incorrect,
                error_rate=incorrect / total if total else 0.0,
            )
        return metrics

    def error_distribution(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[tuple[int, int], int]:
        """Count each (true, predicted) misclassification pair.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Mapping of ``(true_label, predicted_label)`` tuples to counts.
        """
        _validate(y_true, y_pred)
        distribution: dict[tuple[int, int], int] = {}
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                key = (int(true_label), int(pred_label))
                distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> ErrorAnalysisResult:
        """Run a complete error analysis.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            An :class:`ErrorAnalysisResult` with overall and per-class metrics.
        """
        _validate(y_true, y_pred)
        indices = self.error_indices(y_true, y_pred)
        total = len(y_true)
        error_count = len(indices)
        return ErrorAnalysisResult(
            total_samples=total,
            error_count=error_count,
            error_rate=error_count / total if total else 0.0,
            accuracy=1.0 - (error_count / total if total else 0.0),
            error_indices=indices,
            class_metrics=self.error_rate_by_class(y_true, y_pred),
            error_distribution=self.error_distribution(y_true, y_pred),
        )
