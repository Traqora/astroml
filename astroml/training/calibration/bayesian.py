"""Bayesian-style uncertainty estimation for trained models.

Provides model-agnostic uncertainty estimators: Monte Carlo dropout,
predictive entropy / variance, mutual information, and bootstrap
ensembles. The helpers operate on prediction matrices rather than on a
specific framework, so they work with any model exposing a predict
callable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_EPS = 1e-12


class BayesianUncertainty:
    """Estimate predictive uncertainty using Bayesian approximations."""

    def monte_carlo_dropout(
        self,
        model: Any,
        X: np.ndarray,
        predict_fn: Callable[[Any, np.ndarray], np.ndarray],
        n_iterations: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate uncertainty via Monte Carlo dropout.

        Runs the model ``n_iterations`` times with dropout active and
        aggregates the predictions.

        Args:
            model: The trained model with dropout enabled at inference.
            X: Input samples.
            predict_fn: Callable mapping ``(model, X)`` to probability vectors.
            n_iterations: Number of stochastic forward passes.

        Returns:
            Tuple of (mean predictions, standard deviation of predictions).

        Raises:
            ValueError: If ``n_iterations`` is not positive.
        """
        if n_iterations < 1:
            raise ValueError("n_iterations must be positive")
        samples = np.stack(
            [np.asarray(predict_fn(model, X), dtype=float) for _ in range(n_iterations)]
        )
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        return mean, std

    def predictive_entropy(self, probs: np.ndarray) -> np.ndarray:
        """Compute predictive entropy per sample.

        Args:
            probs: Probability vectors, shape ``(n, n_classes)``.

        Returns:
            Per-sample predictive entropy.

        Raises:
            ValueError: If probabilities are outside ``[0, 1]``.
        """
        probs = self._validate_probs(probs)
        return -np.sum(probs * np.log(probs + _EPS), axis=1)

    def predictive_variance(self, probs: np.ndarray) -> np.ndarray:
        """Compute predictive variance per sample.

        Args:
            probs: Probability vectors, shape ``(n, n_classes)``.

        Returns:
            Per-sample variance of the probability distribution.
        """
        probs = self._validate_probs(probs)
        return np.sum(probs * (1.0 - probs), axis=1)

    def mutual_information(self, samples: np.ndarray) -> np.ndarray:
        """Compute mutual information across an ensemble of predictions.

        ``MI = H(E[p]) - E[H(p)]`` is high when ensemble members disagree.

        Args:
            samples: Stacked probability vectors, shape
                ``(n_members, n, n_classes)``.

        Returns:
            Per-sample mutual information.

        Raises:
            ValueError: If the input is not 3-dimensional.
        """
        if samples.ndim != 3:
            raise ValueError("samples must be a 3D array (n_members, n, n_classes)")
        mean_probs = samples.mean(axis=0)
        entropy_of_mean = self.predictive_entropy(mean_probs)
        mean_of_entropy = np.mean(
            [self.predictive_entropy(samples[i]) for i in range(samples.shape[0])], axis=0
        )
        return entropy_of_mean - mean_of_entropy

    def bootstrap_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        n_bootstraps: int = 20,
        sample_fraction: float = 0.8,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate uncertainty with bootstrap-aggregated models.

        Args:
            X: Feature matrix.
            y: Target labels.
            fit_predict_fn: Callable ``(X_train, y_train, X_test) -> predictions``.
            n_bootstraps: Number of bootstrap resamples.
            sample_fraction: Fraction of samples in each bootstrap.
            random_state: Seed for reproducible resampling.

        Returns:
            Tuple of (mean predictions, standard deviation of predictions).

        Raises:
            ValueError: If ``n_bootstraps`` is not positive or the sample
                fraction is outside ``(0, 1]``.
        """
        if n_bootstraps < 1:
            raise ValueError("n_bootstraps must be positive")
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        rng = np.random.default_rng(random_state)
        X = np.asarray(X)
        n = len(X)
        n_sample = max(1, int(n * sample_fraction))
        predictions = []
        for _ in range(n_bootstraps):
            indices = rng.integers(0, n, size=n_sample)
            preds = np.asarray(fit_predict_fn(X[indices], y[indices], X), dtype=float)
            predictions.append(preds)
        stacked = np.stack(predictions)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def _validate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Validate a probability matrix.

        Args:
            probs: Probability vectors.

        Returns:
            The validated array as float.

        Raises:
            ValueError: If values are outside ``[0, 1]``.
        """
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 2:
            raise ValueError("probs must be a 2D array of shape (n, n_classes)")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError("probs must contain values between 0 and 1")
        return probs
