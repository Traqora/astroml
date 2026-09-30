from astroml.utils.exceptions import AstroMLError
"""AutoML model selection pipeline.

Searches a pool of candidate models under computational budget
constraints, ranks them with cross-validation, and fits the winner on the
full dataset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from astroml.training.model_selection.benchmark import BenchmarkResult, ModelBenchmark

DEFAULT_CLASSIFIERS: dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "svm": SVC(probability=True, random_state=42),
    "knn": KNeighborsClassifier(),
}


@dataclass
class AutoMLConfig:
    """Configuration for an AutoML search.

    Attributes:
        task: Task type; only "classification" is currently supported.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for ranking candidates.
        max_models: Maximum number of candidates to evaluate.
        time_budget: Maximum search time in seconds (0 disables the limit).
        random_state: Seed for reproducible splits.
    """

    task: str = "classification"
    cv: int = 5
    scoring: str = "accuracy"
    max_models: int = 6
    time_budget: float = 0.0
    random_state: int = 42


@dataclass
class AutoMLResult:
    """Result of an AutoML search.

    Attributes:
        best_model_name: Name of the winning model.
        best_score: CV score of the winning model.
        leaderboard: Ranked benchmark results.
        params: Parameters of the winning model.
        fitted_model: The winning model fitted on the full dataset.
        searched: Number of candidates actually evaluated.
    """

    best_model_name: str
    best_score: float
    leaderboard: list[BenchmarkResult] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    fitted_model: Any | None = None
    searched: int = 0


class AutoMLPipeline:
    """Run automated model selection with budget constraints."""

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: AutoMLConfig | None = None,
        models: dict[str, Any] | None = None,
    ) -> AutoMLResult:
        """Search for the best model among the candidates.

        Args:
            X: Feature matrix.
            y: Target labels.
            config: Search configuration; defaults to :class:`AutoMLConfig`.
            models: Candidate models keyed by name; defaults to the built-in
                classifier pool.

        Returns:
            An :class:`AutoMLResult` with the winning model.

        Raises:
            ValueError: If the task is unsupported or inputs are invalid.
        """
        config = config or AutoMLConfig()
        if config.task != "classification":
            raise ValueError(f"Unsupported task type: {config.task!r}")
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have matching lengths")

        candidates = models if models is not None else dict(DEFAULT_CLASSIFIERS)
        candidates = dict(list(candidates.items())[: config.max_models])
        if not candidates:
            raise ValueError("No candidate models available")

        benchmark = ModelBenchmark()
        results: list[BenchmarkResult] = []
        searched = 0
        start = time.perf_counter()
        for name, model in candidates.items():
            if self._budget_exceeded(config.time_budget, start):
                break
            try:
                result = benchmark.run(
                    {name: model},
                    X,
                    y,
                    cv=config.cv,
                    scoring=config.scoring,
                    random_state=config.random_state,
                )[0]
                results.append(result)
                searched += 1
            except AstroMLError:
                continue

        ranked = benchmark.compare(results)
        if not ranked:
            raise ValueError("All candidate models failed to evaluate")
        best = ranked[0]

        best_model = candidates[best.model_name]
        best_model.fit(X, y)
        return AutoMLResult(
            best_model_name=best.model_name,
            best_score=best.cv_mean,
            leaderboard=ranked,
            params=best.params,
            fitted_model=best_model,
            searched=searched,
        )

    def predict(self, result: AutoMLResult, X: np.ndarray) -> np.ndarray:
        """Predict with the fitted best model.

        Args:
            result: The AutoML result containing the fitted model.
            X: Feature matrix.

        Returns:
            Predicted labels.

        Raises:
            ValueError: If the result has no fitted model.
        """
        if result.fitted_model is None:
            raise ValueError("AutoMLResult has no fitted model; run search() first")
        return result.fitted_model.predict(X)

    def predict_proba(self, result: AutoMLResult, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities with the fitted best model.

        Args:
            result: The AutoML result containing the fitted model.
            X: Feature matrix.

        Returns:
            Predicted probabilities.

        Raises:
            ValueError: If the result has no fitted model.
        """
        if result.fitted_model is None:
            raise ValueError("AutoMLResult has no fitted model; run search() first")
        predict_proba = getattr(result.fitted_model, "predict_proba", None)
        if predict_proba is None:
            raise ValueError("The selected model does not support predict_proba")
        return predict_proba(X)

    def _budget_exceeded(self, time_budget: float, start: float) -> bool:
        """Check whether the time budget has been exceeded.

        Args:
            time_budget: Budget in seconds; 0 disables the check.
            start: Timestamp when the search started.

        Returns:
            True if the budget was exceeded, False otherwise.
        """
        return time_budget > 0 and (time.perf_counter() - start) >= time_budget
