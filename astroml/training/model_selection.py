"""Model selection and hyperparameter optimization utilities for AstroML."""

from __future__ import annotations

import copy
import itertools
import logging
from collections.abc import Generator, Mapping, Sequence
from typing import Any

import numpy as np

from astroml.validation.cross_validation import (
    CrossValidationReport,
    CrossValidator,
    cross_validate,
)
from astroml.validation.splitters import (
    BaseSplitter,
    GroupKFoldSplitter,
    KFoldSplitter,
    PurgedWalkForwardSplitter,
    SlidingWindowSplitter,
    SplitterConfig,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    _to_numpy_array,
    get_splitter,
)

logger = logging.getLogger(__name__)


class GridSearchCV:
    """Exhaustive search over specified parameter values for an estimator.

    Parameters
    ----------
    estimator : Any
        Object implementing fit and predict.
    param_grid : dict[str, Sequence[Any]] | list[dict[str, Sequence[Any]]]
        Dictionary with parameter names as keys and lists of parameter settings to try.
    cv : BaseSplitter | int
        Cross-validation splitting strategy.
    scoring : str
        Metric name to optimize ('accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'mse', 'r2').
    refit : bool
        Whether to refit the best estimator using the best found parameters on the whole dataset.
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: Mapping[str, Sequence[Any]] | list[Mapping[str, Sequence[Any]]],
        cv: BaseSplitter | int = 5,
        scoring: str = "accuracy",
        refit: bool = True,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

        self.cv_results_: dict[str, Any] = {}
        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = float("-inf")
        self.best_estimator_: Any = None
        self.best_index_: int = -1

    def _generate_param_candidates(self) -> list[dict[str, Any]]:
        grids = [self.param_grid] if isinstance(self.param_grid, dict) else self.param_grid
        candidates = []
        for g in grids:
            keys = list(g.keys())
            values = list(g.values())
            for combination in itertools.product(*values):
                candidates.append(dict(zip(keys, combination)))
        return candidates

    def fit(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        timestamps: Any = None,
    ) -> GridSearchCV:
        """Run fit with all sets of parameters."""
        candidates = self._generate_param_candidates()
        if not candidates:
            raise ValueError("Parameter grid is empty")

        mean_scores = []
        std_scores = []
        all_split_scores = []

        is_higher_better = self.scoring not in {
            "mse",
            "mean_squared_error",
            "mae",
            "mean_absolute_error",
            "log_loss",
        }
        best_score = float("-inf") if is_higher_better else float("inf")
        best_candidate = candidates[0]
        best_idx = 0

        for idx, params in enumerate(candidates):
            candidate_model = copy.deepcopy(self.estimator)
            for k, v in params.items():
                if hasattr(candidate_model, "set_params"):
                    candidate_model.set_params(**{k: v})
                elif hasattr(candidate_model, k):
                    setattr(candidate_model, k, v)

            report = cross_validate(
                estimator=candidate_model,
                X=X,
                y=y,
                groups=groups,
                timestamps=timestamps,
                cv=self.cv,
                scoring=self.scoring,
            )

            metric_stats = report.metrics_summary.get(self.scoring, {})
            m_score = metric_stats.get("mean", 0.0)
            s_score = metric_stats.get("std", 0.0)

            mean_scores.append(m_score)
            std_scores.append(s_score)
            all_split_scores.append([fold.get(self.scoring, 0.0) for fold in report.test_scores])

            if (is_higher_better and m_score > best_score) or (
                not is_higher_better and m_score < best_score
            ):
                best_score = m_score
                best_candidate = params
                best_idx = idx

        self.best_params_ = best_candidate
        self.best_score_ = best_score
        self.best_index_ = best_idx

        self.cv_results_ = {
            "params": candidates,
            "mean_test_score": np.array(mean_scores),
            "std_test_score": np.array(std_scores),
            "split_test_scores": all_split_scores,
        }

        if self.refit:
            self.best_estimator_ = copy.deepcopy(self.estimator)
            for k, v in self.best_params_.items():
                if hasattr(self.best_estimator_, "set_params"):
                    self.best_estimator_.set_params(**{k: v})
                elif hasattr(self.best_estimator_, k):
                    setattr(self.best_estimator_, k, v)
            if y is not None:
                self.best_estimator_.fit(X, y)
            else:
                self.best_estimator_.fit(X)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Call predict on the estimator with the best found parameters."""
        if self.best_estimator_ is None:
            raise RuntimeError("GridSearchCV must be fitted before predict")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Call predict_proba on the estimator with the best found parameters."""
        if self.best_estimator_ is None:
            raise RuntimeError("GridSearchCV must be fitted before predict_proba")
        return self.best_estimator_.predict_proba(X)


class RandomizedSearchCV(GridSearchCV):
    """Randomized search on hyper parameters.

    Parameters
    ----------
    estimator : Any
    param_distributions : dict[str, Sequence[Any]]
    n_iter : int
        Number of parameter settings sampled (default 10).
    cv : BaseSplitter | int
    scoring : str
    random_state : int | None
    refit : bool
    """

    def __init__(
        self,
        estimator: Any,
        param_distributions: Mapping[str, Sequence[Any]],
        n_iter: int = 10,
        cv: BaseSplitter | int = 5,
        scoring: str = "accuracy",
        random_state: int | None = None,
        refit: bool = True,
    ) -> None:
        super().__init__(
            estimator=estimator,
            param_grid=param_distributions,
            cv=cv,
            scoring=scoring,
            refit=refit,
        )
        self.n_iter = n_iter
        self.random_state = random_state

    def _generate_param_candidates(self) -> list[dict[str, Any]]:
        all_candidates = super()._generate_param_candidates()
        if len(all_candidates) <= self.n_iter:
            return all_candidates

        rng = np.random.default_rng(self.random_state)
        chosen_indices = rng.choice(len(all_candidates), size=self.n_iter, replace=False)
        return [all_candidates[i] for i in chosen_indices]


def evaluate_model_cv(
    estimator: Any,
    X: Any,
    y: Any = None,
    splitter_type: str = "time_series",
    n_splits: int = 5,
    scoring: Sequence[str] | str | None = None,
    timestamps: Any = None,
    **kwargs: Any,
) -> CrossValidationReport:
    """Convenience helper to evaluate a model with a configured cross-validator."""
    splitter = get_splitter(splitter_type, n_splits=n_splits, **kwargs)
    return cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        timestamps=timestamps,
        cv=splitter,
        scoring=scoring,
    )


__all__ = [
    "GridSearchCV",
    "RandomizedSearchCV",
    "evaluate_model_cv",
    "cross_validate",
    "CrossValidator",
    "CrossValidationReport",
    "BaseSplitter",
    "KFoldSplitter",
    "StratifiedKFoldSplitter",
    "GroupKFoldSplitter",
    "TimeSeriesSplitter",
    "SlidingWindowSplitter",
    "PurgedWalkForwardSplitter",
    "SplitterConfig",
    "get_splitter",
]
