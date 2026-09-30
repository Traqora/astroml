"""Platt scaling for probability calibration.

Platt scaling fits a logistic regression on the logits of a model's raw
scores to produce calibrated probabilities. The implementation is
scikit-learn based and exposes the learned ``a``/``b`` parameters for
inspection.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-7


class PlattCalibrator:
    """Calibrate binary probabilities using Platt scaling.

    Attributes:
        a: Learned scale parameter of the logistic model.
        b: Learned offset parameter of the logistic model.
        fitted: Whether the calibrator has been fitted.
    """

    def __init__(self, max_iter: int = 1000, c: float = 1e6) -> None:
        """Initialize the calibrator.

        Args:
            max_iter: Maximum iterations for the underlying logistic regression.
            c: Regularization strength; a large value approximates unregularized
                logistic regression as used in classic Platt scaling.
        """
        self.max_iter = max_iter
        self.c = c
        self.a: float = 1.0
        self.b: float = 0.0
        self.fitted = False
        self._model: LogisticRegression | None = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        """Fit the calibration model.

        Args:
            y_prob: Raw predicted probabilities in ``[0, 1]``.
            y_true: Ground truth binary labels (0/1).

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If inputs are misaligned, empty, or contain invalid
                probabilities.
        """
        self._validate(y_prob, y_true)
        logits = self._to_logits(y_prob)
        model = LogisticRegression(max_iter=self.max_iter, C=self.c)
        model.fit(logits.reshape(-1, 1), y_true)
        self.a = float(model.coef_[0][0])
        self.b = float(model.intercept_[0])
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
            raise RuntimeError("PlattCalibrator must be fitted before calibrate()")
        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError("y_prob must be between 0 and 1")
        logits = self._to_logits(y_prob)
        calibrated = self._model.predict_proba(logits.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, _EPS, 1.0 - _EPS)

    def calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling directly to raw scores (logits).

        Args:
            scores: Raw model scores (logits).

        Returns:
            Calibrated probabilities via the sigmoid ``1 / (1 + exp(a*x + b))``.
        """
        if not self.fitted:
            raise RuntimeError("PlattCalibrator must be fitted before calibrate_scores()")
        return 1.0 / (1.0 + np.exp(-(self.a * scores + self.b)))

    def _to_logits(self, y_prob: np.ndarray) -> np.ndarray:
        """Convert probabilities to logits, clipping extreme values.

        Args:
            y_prob: Probabilities in ``[0, 1]``.

        Returns:
            Logits computed as ``log(p / (1 - p))``.
        """
        clipped = np.clip(y_prob, _EPS, 1.0 - _EPS)
        return np.log(clipped / (1.0 - clipped))

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
            raise ValueError("At least two samples are required for Platt scaling")
        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError("y_prob must be between 0 and 1")
        if not set(np.unique(y_true)).issubset({0, 1}):
            raise ValueError("y_true must contain only binary labels (0/1)")
