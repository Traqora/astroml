"""Failure mode identification for classification models.

Identifies structured failure modes from model predictions, such as false
positives, false negatives, high-confidence errors and slice-specific
failures, so teams can prioritize debugging effort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Severity = str  # one of "low", "medium", "high"


@dataclass
class FailureMode:
    """A single identified failure mode.

    Attributes:
        name: Machine-readable identifier of the failure mode.
        description: Human-readable explanation of the failure mode.
        sample_indices: Indices of samples affected by this failure mode.
        count: Number of affected samples.
        rate: Affected samples as a fraction of all samples.
        severity: "low", "medium" or "high" based on the affected rate.
    """

    name: str
    description: str
    sample_indices: list[int]
    count: int
    rate: float
    severity: Severity


@dataclass
class FailureModeReport:
    """Report of all identified failure modes.

    Attributes:
        modes: Identified failure modes.
        total_errors: Total number of misclassified samples.
    """

    modes: list[FailureMode] = field(default_factory=list)
    total_errors: int = 0


class FailureModeIdentifier:
    """Identify common failure modes in a model's predictions."""

    def __init__(self, confidence_threshold: float = 0.8) -> None:
        """Initialize the identifier.

        Args:
            confidence_threshold: Confidence above which an error is
                considered a high-confidence error.
        """
        self.confidence_threshold = confidence_threshold

    def identify(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        slice_labels: np.ndarray | None = None,
    ) -> FailureModeReport:
        """Identify failure modes from predictions.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_prob: Optional predicted probabilities of the predicted class.
            slice_labels: Optional slice identifier per sample; enables
                slice-specific failure modes.

        Returns:
            A :class:`FailureModeReport`.

        Raises:
            ValueError: If the input arrays are misaligned or empty.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}"
            )
        if len(y_true) == 0:
            raise ValueError("y_true and y_pred must not be empty")
        if y_prob is not None and y_prob.shape != y_pred.shape:
            raise ValueError(
                f"y_prob shape {y_prob.shape} does not match y_pred shape {y_pred.shape}"
            )

        total = len(y_true)
        errors = y_true != y_pred
        total_errors = int(np.count_nonzero(errors))

        report = FailureModeReport(total_errors=total_errors)
        report.modes.append(self._false_positives(y_true, y_pred))
        report.modes.append(self._false_negatives(y_true, y_pred))
        if y_prob is not None:
            report.modes.append(self._high_confidence_errors(errors, y_prob))
        if slice_labels is not None:
            report.modes.append(self._slice_failures(y_true, y_pred, slice_labels))
        return report

    def summarize(self, report: FailureModeReport) -> dict[str, object]:
        """Produce a compact summary of a failure mode report.

        Args:
            report: The failure mode report to summarize.

        Returns:
            Dictionary with mode names, counts and severities.
        """
        return {
            "total_errors": report.total_errors,
            "modes": [
                {
                    "name": mode.name,
                    "count": mode.count,
                    "rate": mode.rate,
                    "severity": mode.severity,
                    "description": mode.description,
                }
                for mode in report.modes
            ],
        }

    def _false_positives(self, y_true: np.ndarray, y_pred: np.ndarray) -> FailureMode:
        """Build the false positive failure mode.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            A :class:`FailureMode` describing false positives.
        """
        indices = np.flatnonzero((y_pred == 1) & (y_true == 0))
        return self._make_mode(
            name="false_positives",
            description="Samples predicted as positive that are actually negative.",
            indices=indices,
            total=len(y_true),
        )

    def _false_negatives(self, y_true: np.ndarray, y_pred: np.ndarray) -> FailureMode:
        """Build the false negative failure mode.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            A :class:`FailureMode` describing false negatives.
        """
        indices = np.flatnonzero((y_pred == 0) & (y_true == 1))
        return self._make_mode(
            name="false_negatives",
            description="Samples predicted as negative that are actually positive.",
            indices=indices,
            total=len(y_true),
        )

    def _high_confidence_errors(self, errors: np.ndarray, y_prob: np.ndarray) -> FailureMode:
        """Build the high-confidence error failure mode.

        Args:
            errors: Boolean mask of misclassified samples.
            y_prob: Predicted probabilities of the predicted class.

        Returns:
            A :class:`FailureMode` describing high-confidence errors.
        """
        indices = np.flatnonzero(errors & (y_prob >= self.confidence_threshold))
        return self._make_mode(
            name="high_confidence_errors",
            description=(
                f"Misclassified samples with confidence >= {self.confidence_threshold}. "
                "These are especially surprising and worth investigating first."
            ),
            indices=indices,
            total=len(errors),
        )

    def _slice_failures(
        self, y_true: np.ndarray, y_pred: np.ndarray, slice_labels: np.ndarray
    ) -> FailureMode:
        """Build a failure mode for slices with above-average error rates.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            slice_labels: Slice identifier per sample.

        Returns:
            A :class:`FailureMode` describing slice-specific failures.
        """
        if slice_labels.shape != y_true.shape:
            raise ValueError(
                f"slice_labels shape {slice_labels.shape} does not match y_true shape {y_true.shape}"
            )
        overall_error = float(np.mean(y_true != y_pred))
        failing_mask = np.zeros(len(y_true), dtype=bool)
        for slice_name in np.unique(slice_labels):
            mask = slice_labels == slice_name
            slice_error = float(np.mean(y_true[mask] != y_pred[mask]))
            if slice_error > overall_error:
                failing_mask |= mask
        indices = np.flatnonzero(failing_mask)
        return self._make_mode(
            name="slice_failures",
            description="Samples belonging to slices whose error rate exceeds the overall error rate.",
            indices=indices,
            total=len(y_true),
        )

    def _make_mode(
        self, name: str, description: str, indices: np.ndarray, total: int
    ) -> FailureMode:
        """Construct a failure mode with computed rate and severity.

        Args:
            name: Machine-readable identifier of the failure mode.
            description: Human-readable description.
            indices: Affected sample indices.
            total: Total number of samples.

        Returns:
            A fully populated :class:`FailureMode`.
        """
        count = len(indices)
        rate = count / total if total else 0.0
        severity = self._severity(rate)
        return FailureMode(
            name=name,
            description=description,
            sample_indices=indices.tolist(),
            count=count,
            rate=rate,
            severity=severity,
        )

    def _severity(self, rate: float) -> Severity:
        """Map a failure rate to a severity level.

        Args:
            rate: Fraction of samples affected.

        Returns:
            "high" for rates >= 0.2, "medium" for >= 0.05, else "low".
        """
        if rate >= 0.2:
            return "high"
        if rate >= 0.05:
            return "medium"
        return "low"
