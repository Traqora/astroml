"""Split conformal prediction for classification and regression.

Conformal prediction produces prediction sets (classification) or
prediction intervals (regression) with finite-sample coverage guarantees,
using only a held-out calibration set — no retraining required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformalResult:
    """Output of a conformal prediction step.

    Attributes:
        prediction_sets: Per-sample sets of class labels (classification).
        quantile: The calibrated nonconformity quantile used.
        coverage: Empirical coverage on the provided test labels, if any.
        lower_bounds: Per-sample interval lower bounds (regression).
        upper_bounds: Per-sample interval upper bounds (regression).
        classes: Class labels corresponding to prediction set indices.
    """

    prediction_sets: list[list[int]] = field(default_factory=list)
    quantile: float = 0.0
    coverage: float | None = None
    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    classes: list[int] = field(default_factory=list)


class ConformalPredictor:
    """Split conformal predictor with classification and regression modes."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize the predictor.

        Args:
            random_state: Seed for tie-breaking in quantile computation.
        """
        self.random_state = random_state
        self._cal_scores: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._classes: np.ndarray | None = None

    def fit_classification(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalPredictor":
        """Fit using a calibration set of class probabilities.

        Args:
            probs_cal: Calibration probabilities, shape ``(n, n_classes)``.
            y_cal: Calibration ground truth labels.

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If the inputs are invalid.
        """
        probs_cal, y_cal = self._validate_probs(probs_cal, y_cal)
        self._classes = np.unique(y_cal)
        class_index = {int(label): idx for idx, label in enumerate(self._classes)}
        scores = np.empty(len(y_cal))
        for i, label in enumerate(y_cal):
            idx = class_index[int(label)]
            scores[i] = 1.0 - probs_cal[i, idx]
        self._cal_scores = scores
        self._residuals = None
        return self

    def fit_regression(self, y_pred_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalPredictor":
        """Fit using calibration residuals for regression.

        Args:
            y_pred_cal: Calibration point predictions.
            y_cal: Calibration ground truth targets.

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If the inputs are misaligned or empty.
        """
        y_pred_cal = np.asarray(y_pred_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)
        if y_pred_cal.shape != y_cal.shape:
            raise ValueError(
                f"y_pred_cal shape {y_pred_cal.shape} does not match y_cal shape {y_cal.shape}"
            )
        if len(y_cal) == 0:
            raise ValueError("Calibration arrays must not be empty")
        self._residuals = np.abs(y_pred_cal - y_cal)
        self._cal_scores = None
        return self

    def predict_sets(
        self,
        probs: np.ndarray,
        significance: float = 0.1,
        y_true: np.ndarray | None = None,
    ) -> ConformalResult:
        """Build conformal prediction sets for classification.

        Args:
            probs: Predicted class probabilities, shape ``(n, n_classes)``.
            significance: Desired miscoverage level ``alpha``.
            y_true: Optional true labels to compute empirical coverage.

        Returns:
            A :class:`ConformalResult` with prediction sets.

        Raises:
            RuntimeError: If the predictor was not fitted for classification.
            ValueError: If inputs are invalid.
        """
        if self._cal_scores is None:
            raise RuntimeError("ConformalPredictor must be fitted with fit_classification() first")
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1")
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 2:
            raise ValueError("probs must be a 2D array of shape (n, n_classes)")

        quantile = self._score_quantile(significance)
        threshold = 1.0 - quantile
        prediction_sets: list[list[int]] = []
        for row in probs:
            labels = [
                int(self._classes[i]) for i in range(len(self._classes)) if row[i] >= threshold
            ]
            prediction_sets.append(labels)

        coverage = self._compute_coverage(probs, y_true, threshold)
        return ConformalResult(
            prediction_sets=prediction_sets,
            quantile=quantile,
            coverage=coverage,
            classes=[int(label) for label in self._classes],
        )

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        significance: float = 0.1,
        y_true: np.ndarray | None = None,
    ) -> ConformalResult:
        """Build conformal prediction intervals for regression.

        Args:
            y_pred: Point predictions for new samples.
            significance: Desired miscoverage level ``alpha``.
            y_true: Optional true targets to compute empirical coverage.

        Returns:
            A :class:`ConformalResult` with interval bounds.

        Raises:
            RuntimeError: If the predictor was not fitted for regression.
            ValueError: If inputs are invalid.
        """
        if self._residuals is None:
            raise RuntimeError("ConformalPredictor must be fitted with fit_regression() first")
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1")
        y_pred = np.asarray(y_pred, dtype=float)
        quantile = self._score_quantile(significance)
        lower = y_pred - quantile
        upper = y_pred + quantile

        coverage: float | None = None
        if y_true is not None:
            y_true = np.asarray(y_true, dtype=float)
            if y_true.shape != y_pred.shape:
                raise ValueError("y_true must match y_pred in shape")
            contained = (y_true >= lower) & (y_true <= upper)
            coverage = float(np.mean(contained))

        return ConformalResult(
            quantile=quantile,
            coverage=coverage,
            lower_bounds=lower.tolist(),
            upper_bounds=upper.tolist(),
        )

    def _score_quantile(self, significance: float) -> float:
        """Compute the calibrated nonconformity quantile.

        Args:
            significance: Desired miscoverage level ``alpha``.

        Returns:
            The ``ceil((n + 1) * (1 - alpha)) / n`` quantile of scores.
        """
        scores = self._cal_scores if self._cal_scores is not None else self._residuals
        assert scores is not None
        n = len(scores)
        level = min(1.0, float(np.ceil((n + 1) * (1 - significance))) / n)
        rng = np.random.default_rng(self.random_state)
        return float(np.quantile(scores, level, method="higher"))

    def _compute_coverage(
        self,
        probs: np.ndarray,
        y_true: np.ndarray | None,
        threshold: float,
    ) -> float | None:
        """Compute empirical coverage of prediction sets.

        Args:
            probs: Predicted class probabilities.
            y_true: True labels, or None.
            threshold: Probability threshold for set membership.

        Returns:
            Empirical coverage fraction, or None if labels are missing.
        """
        if y_true is None:
            return None
        y_true = np.asarray(y_true)
        if y_true.shape[0] != probs.shape[0]:
            raise ValueError("y_true must match probs in the number of samples")
        class_index = {int(label): idx for idx, label in enumerate(self._classes)}
        covered = 0
        for i, label in enumerate(y_true):
            idx = class_index[int(label)]
            if probs[i, idx] >= threshold:
                covered += 1
        return covered / len(y_true)

    def _validate_probs(
        self, probs: np.ndarray, y_cal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate classification calibration inputs.

        Args:
            probs: Calibration probabilities.
            y_cal: Calibration labels.

        Returns:
            The validated arrays.

        Raises:
            ValueError: If the inputs are invalid.
        """
        probs = np.asarray(probs, dtype=float)
        y_cal = np.asarray(y_cal)
        if probs.ndim != 2:
            raise ValueError("probs must be a 2D array of shape (n, n_classes)")
        if probs.shape[0] != y_cal.shape[0]:
            raise ValueError("probs and y_cal must have the same number of samples")
        if probs.shape[0] == 0:
            raise ValueError("Calibration arrays must not be empty")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("probs must contain values between 0 and 1")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError("probs rows must sum to 1")
        return probs, y_cal
