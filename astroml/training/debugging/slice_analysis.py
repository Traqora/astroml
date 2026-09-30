"""Slice-based performance analysis for classification models.

Evaluates model accuracy across user-defined data slices (for example by
account tier, geography, or feature bucket) and flags slices that
underperform relative to a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SliceMetrics:
    """Performance metrics for a single data slice.

    Attributes:
        slice_name: Label of the slice.
        support: Number of samples in the slice.
        correct: Number of correctly classified samples.
        incorrect: Number of misclassified samples.
        accuracy: Accuracy within the slice.
        error_rate: Error rate within the slice.
        precision: Precision within the slice (binary, label 1 as positive).
        recall: Recall within the slice (binary, label 1 as positive).
        f1: F1 score within the slice (binary, label 1 as positive).
    """

    slice_name: str
    support: int
    correct: int
    incorrect: int
    accuracy: float
    error_rate: float
    precision: float
    recall: float
    f1: float


@dataclass
class SliceAnalysisResult:
    """Result of a slice-based performance analysis.

    Attributes:
        slices: Metrics for each analyzed slice.
        overall_accuracy: Accuracy across all samples.
        worst_slice: Name of the slice with the lowest accuracy.
        worst_accuracy: Accuracy of the worst slice.
    """

    slices: list[SliceMetrics] = field(default_factory=list)
    overall_accuracy: float = 0.0
    worst_slice: str | None = None
    worst_accuracy: float = 1.0


class SliceAnalyzer:
    """Analyze model performance across data slices."""

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        slice_labels: np.ndarray,
    ) -> SliceAnalysisResult:
        """Compute performance metrics for each slice.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            slice_labels: Slice identifier for every sample.

        Returns:
            A :class:`SliceAnalysisResult`.

        Raises:
            ValueError: If the input arrays are misaligned or empty.
        """
        if not (len(y_true) == len(y_pred) == len(slice_labels)):
            raise ValueError("y_true, y_pred and slice_labels must have the same length")
        if len(y_true) == 0:
            raise ValueError("Input arrays must not be empty")

        result = SliceAnalysisResult(overall_accuracy=float(np.mean(y_true == y_pred)))
        for slice_name in np.unique(slice_labels):
            mask = slice_labels == slice_name
            metrics = self._slice_metrics(str(slice_name), y_true[mask], y_pred[mask])
            result.slices.append(metrics)
            if metrics.accuracy < result.worst_accuracy:
                result.worst_accuracy = metrics.accuracy
                result.worst_slice = metrics.slice_name
        return result

    def underperforming_slices(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        slice_labels: np.ndarray,
        threshold: float = 0.6,
    ) -> list[SliceMetrics]:
        """Return slices whose accuracy falls below a threshold.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            slice_labels: Slice identifier for every sample.
            threshold: Minimum acceptable accuracy; slices below it are flagged.

        Returns:
            List of :class:`SliceMetrics` for underperforming slices.
        """
        result = self.analyze(y_true, y_pred, slice_labels)
        return [metrics for metrics in result.slices if metrics.accuracy < threshold]

    def _slice_metrics(
        self, slice_name: str, y_true: np.ndarray, y_pred: np.ndarray
    ) -> SliceMetrics:
        """Compute metrics for one slice.

        Args:
            slice_name: Name of the slice.
            y_true: Ground truth labels within the slice.
            y_pred: Predicted labels within the slice.

        Returns:
            A :class:`SliceMetrics` for the slice.
        """
        support = len(y_true)
        correct = int(np.count_nonzero(y_true == y_pred))
        incorrect = support - correct
        accuracy = correct / support if support else 0.0

        tp = int(np.count_nonzero((y_pred == 1) & (y_true == 1)))
        fp = int(np.count_nonzero((y_pred == 1) & (y_true == 0)))
        fn = int(np.count_nonzero((y_pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return SliceMetrics(
            slice_name=slice_name,
            support=support,
            correct=correct,
            incorrect=incorrect,
            accuracy=accuracy,
            error_rate=incorrect / support if support else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
        )
