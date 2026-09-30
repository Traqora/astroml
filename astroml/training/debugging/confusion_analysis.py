"""Confusion matrix analysis for classification models.

Computes raw and normalized confusion matrices together with per-class
derived metrics (precision, recall, F1) so model errors can be inspected
per class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import confusion_matrix

Normalization = str  # one of "true", "pred", "all", "none"


@dataclass
class ClassMetrics:
    """Per-class metrics derived from a confusion matrix.

    Attributes:
        label: The class label.
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        tn: True negatives.
        precision: Precision for this class.
        recall: Recall for this class.
        f1: F1 score for this class.
        support: Number of true samples for this class.
    """

    label: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ConfusionAnalysisResult:
    """Result of a confusion analysis run.

    Attributes:
        labels: Ordered list of class labels.
        matrix: Raw confusion matrix (rows = true, columns = predicted).
        normalized: Row-normalized confusion matrix.
        accuracy: Overall accuracy.
        class_metrics: Per-class metrics keyed by class label.
    """

    labels: list[int]
    matrix: np.ndarray
    normalized: np.ndarray
    accuracy: float
    class_metrics: dict[int, ClassMetrics] = field(default_factory=dict)


class ConfusionAnalyzer:
    """Compute and analyze confusion matrices for classifier predictions."""

    def normalize(self, matrix: np.ndarray, norm: Normalization = "true") -> np.ndarray:
        """Normalize a confusion matrix along rows, columns, or overall.

        Args:
            matrix: Raw confusion matrix.
            norm: Normalization mode: "true", "pred", "all" or "none".

        Returns:
            Normalized matrix. Returns the input unchanged for "none".

        Raises:
            ValueError: If ``norm`` is not a supported mode.
        """
        if norm == "none":
            return matrix
        if norm == "true":
            axis = 1
        elif norm == "pred":
            axis = 0
        elif norm == "all":
            axis = None
        else:
            raise ValueError(f"Unsupported normalization mode: {norm!r}")
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = matrix / matrix.sum(axis=axis, keepdims=True).astype(float)
        return np.nan_to_num(normalized, nan=0.0)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[int] | None = None,
        norm: Normalization = "true",
    ) -> ConfusionAnalysisResult:
        """Compute a confusion matrix and derived per-class metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            labels: Optional ordered class labels; inferred when omitted.
            norm: Normalization mode for the ``normalized`` matrix.

        Returns:
            A :class:`ConfusionAnalysisResult`.

        Raises:
            ValueError: If the input arrays are misaligned or empty.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}"
            )
        if len(y_true) == 0:
            raise ValueError("y_true and y_pred must not be empty")

        resolved_labels = (
            labels if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        )
        matrix = confusion_matrix(y_true, y_pred, labels=resolved_labels)
        normalized = self.normalize(matrix, norm)

        accuracy = float(np.mean(y_true == y_pred))
        class_metrics: dict[int, ClassMetrics] = {}
        for i, label in enumerate(resolved_labels):
            tp = int(matrix[i, i])
            fp = int(matrix[:, i].sum() - tp)
            fn = int(matrix[i, :].sum() - tp)
            tn = int(matrix.sum() - tp - fp - fn)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            class_metrics[label] = ClassMetrics(
                label=label,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=precision,
                recall=recall,
                f1=f1,
                support=tp + fn,
            )

        return ConfusionAnalysisResult(
            labels=[int(label) for label in resolved_labels],
            matrix=matrix,
            normalized=normalized,
            accuracy=accuracy,
            class_metrics=class_metrics,
        )

    def to_dict(self, result: ConfusionAnalysisResult) -> dict[str, object]:
        """Serialize a confusion analysis result to plain dictionaries.

        Args:
            result: The result to serialize.

        Returns:
            A JSON-friendly dictionary representation.
        """
        return {
            "labels": result.labels,
            "matrix": result.matrix.tolist(),
            "normalized": result.normalized.tolist(),
            "accuracy": result.accuracy,
            "class_metrics": {
                str(label): {
                    "label": metrics.label,
                    "tp": metrics.tp,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
                    "tn": metrics.tn,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "support": metrics.support,
                }
                for label, metrics in result.class_metrics.items()
            },
        }
