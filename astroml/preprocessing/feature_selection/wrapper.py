from astroml.utils.exceptions import AstroMLError
"""Wrapper-based feature selection methods.

Implements Recursive Feature Elimination (RFE), forward selection,
and backward elimination with any sklearn-compatible estimator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from astroml.preprocessing.feature_selection.filter import SelectionResult

logger = logging.getLogger(__name__)


class WrapperSelector:
    """Wrapper-based feature selection: RFE, forward selection, backward elimination.

    Uses a model's performance (e.g., accuracy, F1) as the evaluation
    criterion for selecting feature subsets.

    Attributes:
        method: ``rfe``, ``forward``, or ``backward``.
        estimator: Scikit-learn compatible estimator.
        n_features_to_select: Target number of features.
        step: Features to remove/add per iteration.
        scoring: Scoring function or name (default: ``accuracy`` for classifiers,
                 ``r2`` for regressors).
        cv: Number of cross-validation folds.
    """

    SUPPORTED_METHODS = ("rfe", "forward", "backward")

    def __init__(
        self,
        estimator: Any,
        method: str = "rfe",
        n_features_to_select: int = 10,
        step: int = 1,
        scoring: str | Callable[..., float] | None = None,
        cv: int | None = None,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported wrapper method '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )

        self.estimator = estimator
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.scoring = scoring
        self.cv = cv
        self._support: NDArray[np.bool_] | None = None
        self._scores: list[float] | None = None
        self._feature_importances: NDArray[np.float64] | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
        feature_names: list[str] | None = None,
    ) -> WrapperSelector:
        """Fit the wrapper selector.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional feature names.

        Returns:
            Self.
        """
        n_features = X.shape[1]
        self._feature_names = feature_names

        if self.method == "rfe":
            self._fit_rfe(X, y)
        elif self.method == "forward":
            self._fit_forward(X, y)
        elif self.method == "backward":
            self._fit_backward(X, y)

        return self

    def transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Return selected features.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Reduced feature matrix.
        """
        if self._support is None:
            raise RuntimeError("WrapperSelector must be fit before transform")
        return X[:, self._support]

    def fit_transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
        feature_names: list[str] | None = None,
    ) -> NDArray[np.float64]:
        """Fit and transform.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional names.

        Returns:
            Reduced matrix.
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_support(self) -> NDArray[np.bool_]:
        """Boolean mask of selected features.

        Returns:
            Boolean array.
        """
        if self._support is None:
            raise RuntimeError("WrapperSelector must be fit first")
        return self._support

    def get_selection_result(self) -> SelectionResult:
        """Structured selection result.

        Returns:
            SelectionResult.
        """
        if self._support is None:
            raise RuntimeError("WrapperSelector must be fit first")

        indices = np.where(self._support)[0]
        scores = []
        if self._feature_importances is not None:
            scores = [float(self._feature_importances[i]) for i in indices]
        elif self._scores:
            scores = list(self._scores)

        return SelectionResult(
            selector_name=f"wrapper-{self.method}",
            num_features_selected=len(indices),
            num_features_total=len(self._support),
            selected_indices=indices.tolist(),
            scores=scores,
            feature_names=(
                [self._feature_names[i] for i in indices]
                if self._feature_names
                else None
            ),
            metadata={
                "method": self.method,
                "n_features_to_select": self.n_features_to_select,
            },
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fit_rfe(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
    ) -> None:
        """Recursive Feature Elimination."""
        try:
            from sklearn.feature_selection import RFE

            selector = RFE(
                estimator=self.estimator,
                n_features_to_select=self.n_features_to_select,
                step=self.step,
            )
            selector.fit(X, y)
            self._support = selector.support_
            self._feature_importances = np.array(
                selector.ranking_.astype(np.float64)
            )
            # Invert ranking so higher = better for scores
            self._feature_importances = (
                self._feature_importances.max() + 1 - self._feature_importances
            )
            self._scores = self._feature_importances.tolist()
        except ImportError:
            logger.warning("sklearn RFE not available; falling back to backward")
            self._fit_backward(X, y)

    def _fit_forward(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
    ) -> None:
        """Forward selection: greedily add best feature."""
        n_features = X.shape[1]
        n_target = min(self.n_features_to_select, n_features)

        selected: list[int] = []
        remaining = list(range(n_features))
        scores_full = np.zeros(n_features, dtype=np.float64)

        from sklearn.model_selection import cross_val_score

        for _ in range(n_target):
            best_score = -1.0
            best_idx = -1
            for cand in remaining:
                candidate_set = selected + [cand]
                X_sub = X[:, candidate_set]
                try:
                    cv_scores = cross_val_score(
                        self.estimator,
                        X_sub,
                        y,
                        cv=min(self.cv or 3, len(y) // 2) or 3,
                        scoring=self.scoring
                    )
                    score = float(np.mean(cv_scores))
                except AstroMLError:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_idx = cand

            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            scores_full[best_idx] = best_score

        mask = np.zeros(n_features, dtype=bool)
        mask[selected] = True
        self._support = mask
        self._feature_importances = scores_full
        self._scores = [float(scores_full[i]) for i in selected]

    def _fit_backward(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
    ) -> None:
        """Backward elimination: greedily remove worst feature."""
        n_features = X.shape[1]
        n_target = min(self.n_features_to_select, n_features)

        selected = set(range(n_features))
        scores_full = np.ones(n_features, dtype=np.float64)

        from sklearn.model_selection import cross_val_score

        # Baseline score
        try:
            base_scores = cross_val_score(
                self.estimator,
                X,
                y,
                cv=min(self.cv or 3, len(y) // 2) or 3,
                scoring=self.scoring,
            )
            baseline = float(np.mean(base_scores))
        except AstroMLError:
            baseline = 0.5

        while len(selected) > n_target:
            worst_score = baseline + 1.0
            worst_idx = -1
            for cand in list(selected):
                temp = list(selected - {cand})
                if not temp:
                    continue
                X_sub = X[:, temp]
                try:
                    cv_scores = cross_val_score(
                        self.estimator,
                        X_sub,
                        y,
                        cv=min(self.cv or 3, len(y) // 2) or 3,
                        scoring=self.scoring,
                    )
                    score = float(np.mean(cv_scores))
                except AstroMLError:
                    score = baseline
                if score < worst_score:
                    worst_score = score
                    worst_idx = cand

            if worst_idx < 0:
                break
            selected.remove(worst_idx)
            scores_full[worst_idx] = 0.0

        mask = np.zeros(n_features, dtype=bool)
        mask[list(selected)] = True
        self._support = mask
        self._feature_importances = scores_full
        self._scores = [float(scores_full[i]) for i in selected]