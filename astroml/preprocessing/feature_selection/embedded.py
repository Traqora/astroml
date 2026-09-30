"""Embedded feature selection methods.

Implements Lasso (L1 regularization), tree-based importance
(Random Forest, Gradient Boosting), and ElasticNet-based selection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroml.preprocessing.feature_selection.filter import SelectionResult

logger = logging.getLogger(__name__)


class EmbeddedSelector:
    """Embedded feature selection: Lasso, tree-based importance, ElasticNet.

    Attributes:
        method: Selection strategy (``lasso``, ``tree``, ``elasticnet``).
        threshold: Minimum importance score. Features below are dropped.
        k: Maximum number of features to keep.
        alpha: Regularization strength for Lasso/ElasticNet.
        l1_ratio: L1 ratio for ElasticNet (0 = Ridge, 1 = Lasso).
        random_state: Random seed.
    """

    SUPPORTED_METHODS = ("lasso", "tree", "elasticnet")

    def __init__(
        self,
        method: str = "tree",
        threshold: float = 0.01,
        k: int | None = None,
        alpha: float = 0.01,
        l1_ratio: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported embedded method '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.threshold = threshold
        self.k = k
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self._support: NDArray[np.bool_] | None = None
        self._importances: NDArray[np.float64] | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
        feature_names: list[str] | None = None,
    ) -> EmbeddedSelector:
        """Fit the embedded selector.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional feature names.

        Returns:
            Self.
        """
        self._feature_names = feature_names

        if self.method == "lasso":
            self._fit_lasso(X, y)
        elif self.method == "tree":
            self._fit_tree(X, y)
        elif self.method == "elasticnet":
            self._fit_elasticnet(X, y)

        return self

    def transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Return selected feature subset.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Reduced feature matrix.
        """
        if self._support is None:
            raise RuntimeError("EmbeddedSelector must be fit before transform")
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
            raise RuntimeError("EmbeddedSelector must be fit first")
        return self._support

    def get_selection_result(self) -> SelectionResult:
        """Structured selection result.

        Returns:
            SelectionResult.
        """
        if self._support is None:
            raise RuntimeError("EmbeddedSelector must be fit first")

        indices = np.where(self._support)[0]
        scores = (
            [float(self._importances[i]) for i in indices]
            if self._importances is not None
            else []
        )

        return SelectionResult(
            selector_name=f"embedded-{self.method}",
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
                "threshold": self.threshold,
                "k": self.k,
            },
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _fit_lasso(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
    ) -> None:
        """Lasso (L1) feature selection."""
        try:
            from sklearn.linear_model import Lasso

            model = Lasso(
                alpha=self.alpha,
                random_state=self.random_state,
                max_iter=2000,
            )
            model.fit(X, y)
            importances = np.abs(model.coef_.ravel())
        except ImportError:
            logger.warning("sklearn not installed; falling back to variance")
            importances = np.var(X, axis=0)

        self._importances = _normalize_importances(importances)
        self._support = self._build_mask(self._importances)

    def _fit_tree(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
    ) -> None:
        """Tree-based feature importance (Random Forest or Gradient Boosting)."""
        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
            )

            n_unique = len(np.unique(y))
            is_classification = n_unique <= 20 and n_unique < len(y) * 0.05

            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

            model.fit(X, y)
            importances = model.feature_importances_
        except ImportError:
            logger.warning("sklearn not installed; falling back to variance")
            importances = np.var(X, axis=0)

        self._importances = _normalize_importances(importances)
        self._support = self._build_mask(self._importances)

    def _fit_elasticnet(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
    ) -> None:
        """ElasticNet feature selection."""
        try:
            from sklearn.linear_model import ElasticNet

            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
                max_iter=2000,
            )
            model.fit(X, y)
            importances = np.abs(model.coef_.ravel())
        except ImportError:
            logger.warning("sklearn not installed; falling back to variance")
            importances = np.var(X, axis=0)

        self._importances = _normalize_importances(importances)
        self._support = self._build_mask(self._importances)

    def _build_mask(self, importances: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Build boolean mask from importances using threshold and k."""
        mask = importances >= self.threshold
        if self.k is not None and self.k > 0:
            # Keep only top k
            indices = np.argsort(-importances)[: self.k]
            mask = np.zeros(len(importances), dtype=bool)
            mask[indices] = True
        return mask


def _normalize_importances(
    importances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize importances to [0, 1] range.

    Args:
        importances: Raw importance array.

    Returns:
        Normalized importances.
    """
    imp = np.abs(importances)
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
    mx = imp.max()
    if mx > 0:
        imp = imp / mx
    return imp