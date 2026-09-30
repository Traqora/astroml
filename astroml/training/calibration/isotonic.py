"""Isotonic regression for probability calibration.

Isotonic regression fits a non-decreasing step function to the predicted
probabilities, which makes no parametric assumptions about the miscalibration
pattern and is generally more flexible than Platt scaling.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Calibrate probabilities with isotonic regression.

    Attributes:
        fitted: Whether the calibrator has been fitted.
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0) -> None:
        """Initialize the calibrator.

        Args:
            y_min: Minimum output value.
            y_max: Maximum output value.
        """
        self.y_min = y_min
        self.y_max = y_max
        self.fitted = False
        self._model: IsotonicRegression | None = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """Fit the isotonic regression model.

        Args:
            y_prob: Raw predicted probabilities in ``[0, 1]``.
            y_true: Ground truth binary labels (0/1).

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If the inputs are invalid.
        """
        self._validate(y_prob, y_true)
        model = IsotonicRegression(y_min=self.y_min, y_max=self.y_max, out_of_bounds="clip")
        model.fit(y_prob, y_true)
        self._model = model
        self.fitted = True
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply the fitted calibration to new probabilities.

        Args:
            y_prob: Raw predicted probabilities.

        Returns:
            Calibrated probabilities.

        Raises:
            RuntimeError: If the calibrator has not been fitted.
            ValueError: If probabilities are outside ``[0, 1]``.
        """
        if not self.fitted or self._model is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before calibrate()")
        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError("y_prob must be between 0 and 1")
        return self._model.predict(y_prob)

    def _validate(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Validate calibration inputs.

        Args:
            y_prob: Predicted probabilities.
            y_true: Ground truth labels.

        Raises:
            ValueError: If the inputs are invalid.
        """
        if y_prob.shape != y_true.shape:
            raise ValueError(
                f"y_prob shape {y_prob.shape} does not match y_true shape {y_true.shape}"
            )
        if len(y_prob) < 2:
            raise ValueError("At least two samples are required for isotonic calibration")
        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError("y_prob must be between 0 and 1")
        if not set(np.unique(y_true)).issubset({0, 1}):
            raise ValueError("y_true must contain only binary labels (0/1)")
