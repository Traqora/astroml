"""Model benchmarking suite.

Cross-validates candidate models, records fit/predict latency, and ranks
the results so the best model for a task can be selected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single model.

    Attributes:
        model_name: Name of the model.
        cv_mean: Mean cross-validation score.
        cv_std: Standard deviation of cross-validation scores.
        fit_time: Time (seconds) to fit on a train split.
        predict_time: Time (seconds) to predict on a test split.
        params: The model's constructor parameters.
    """

    model_name: str
    cv_mean: float
    cv_std: float
    fit_time: float
    predict_time: float
    params: dict[str, Any] = field(default_factory=dict)


class ModelBenchmark:
    """Benchmark and rank candidate models on a dataset."""

    def run(
        self,
        models: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy",
        random_state: int = 42,
    ) -> list[BenchmarkResult]:
        """Benchmark every candidate model.

        Args:
            models: Mapping of model name to unfitted scikit-learn estimator.
            X: Feature matrix.
            y: Target labels.
            cv: Number of cross-validation folds.
            scoring: Scoring metric passed to ``cross_val_score``.
            random_state: Seed for the timing train/test split.

        Returns:
            A list of :class:`BenchmarkResult` per model.

        Raises:
            ValueError: If no models are provided or the data is invalid.
        """
        if not models:
            raise ValueError("At least one model must be provided")
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have matching lengths")
        if len(X) < max(2, cv):
            raise ValueError("Not enough samples for the requested number of folds")

        results: list[BenchmarkResult] = []
        for name, model in models.items():
            results.append(self._benchmark_one(name, model, X, y, cv, scoring, random_state))
        return results

    def compare(self, results: list[BenchmarkResult]) -> list[BenchmarkResult]:
        """Rank benchmark results by mean CV score, descending.

        Args:
            results: Benchmark results to rank.

        Returns:
            Results sorted by ``cv_mean`` descending.
        """
        return sorted(results, key=lambda r: r.cv_mean, reverse=True)

    def best(self, results: list[BenchmarkResult]) -> BenchmarkResult:
        """Return the best benchmark result.

        Args:
            results: Benchmark results to inspect.

        Returns:
            The result with the highest mean CV score.

        Raises:
            ValueError: If ``results`` is empty.
        """
        if not results:
            raise ValueError("No benchmark results provided")
        return max(results, key=lambda r: r.cv_mean)

    def _benchmark_one(
        self,
        name: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int,
        scoring: str,
        random_state: int,
    ) -> BenchmarkResult:
        """Benchmark a single model.

        Args:
            name: Model name.
            model: Unfitted estimator.
            X: Feature matrix.
            y: Target labels.
            cv: Number of folds.
            scoring: Scoring metric.
            random_state: Seed for the timing split.

        Returns:
            A :class:`BenchmarkResult` for the model.
        """
        estimator = clone(model)
        cv_scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, error_score="raise")

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.25, random_state=random_state
        )
        timed = clone(model)
        start = time.perf_counter()
        timed.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        timed.predict(X_test)
        predict_time = time.perf_counter() - start

        params = getattr(model, "get_params", lambda: {})()
        return BenchmarkResult(
            model_name=name,
            cv_mean=float(np.mean(cv_scores)),
            cv_std=float(np.std(cv_scores)),
            fit_time=fit_time,
            predict_time=predict_time,
            params=params,
        )
