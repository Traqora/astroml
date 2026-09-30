"""Filter-based feature selection methods.

Implements correlation, mutual information, chi-squared, variance threshold,
and ANOVA F-value filter methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of a feature selection operation.

    Attributes:
        selector_name: Name of the selector used.
        num_features_selected: Number of features retained.
        num_features_total: Total input features.
        selected_indices: Indices of selected features.
        scores: Importance scores for each (selected) feature.
        feature_names: Optional feature names.
        metadata: Additional metadata from the selection process.
    """

    selector_name: str
    num_features_selected: int
    num_features_total: int
    selected_indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    feature_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FilterSelector:
    """Filter-based feature selection: correlation, mutual information, chi-squared.

    Attributes:
        method: Selection strategy (``correlation``, ``mutual_info``,
                ``chi2``, ``variance``, ``anova``).
        k: Number of top features to select (if None, all passing threshold).
        threshold: Minimum score for a feature to be kept.
        random_state: Seed for reproducible results.
    """

    SUPPORTED_METHODS = ("correlation", "mutual_info", "chi2", "variance", "anova")

    def __init__(
        self,
        method: str = "mutual_info",
        k: int | None = None,
        threshold: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported filter method '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.k = k
        self.threshold = threshold
        self.random_state = random_state
        self._scores: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64] | None = None,
        feature_names: list[str] | None = None,
    ) -> FilterSelector:
        """Compute feature scores without selecting.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            feature_names: Optional feature names.

        Returns:
            Self (fitted selector).
        """
        n_features = X.shape[1]

        if self.method == "correlation":
            self._scores = _filter_correlation(X, y)
        elif self.method == "mutual_info":
            self._scores = _filter_mutual_info(X, y, self.random_state)
        elif self.method == "chi2":
            self._scores = _filter_chi2(X, y)
        elif self.method == "variance":
            self._scores = _filter_variance(X)
        elif self.method == "anova":
            self._scores = _filter_anova(X, y)
        else:
            self._scores = np.zeros(n_features, dtype=np.float64)

        self._feature_names = feature_names
        return self

    def transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Return the selected feature subset.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Ignored (for API compatibility).

        Returns:
            Reduced feature matrix.
        """
        if self._scores is None:
            raise RuntimeError("FilterSelector must be fit before transform")

        indices = self._select_indices()
        return X[:, indices]

    def fit_transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64] | None = None,
        feature_names: list[str] | None = None,
    ) -> NDArray[np.float64]:
        """Fit and transform in one call.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional feature names.

        Returns:
            Reduced feature matrix.
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_support(self) -> NDArray[np.bool_]:
        """Return a boolean mask of selected features.

        Returns:
            Boolean array of shape (n_features,).
        """
        if self._scores is None:
            raise RuntimeError("FilterSelector must be fit first")
        mask = np.zeros(len(self._scores), dtype=bool)
        mask[self._select_indices()] = True
        return mask

    def get_selection_result(self) -> SelectionResult:
        """Return a structured selection result.

        Returns:
            SelectionResult with indices, scores, and metadata.
        """
        if self._scores is None:
            raise RuntimeError("FilterSelector must be fit first")

        indices = self._select_indices()
        return SelectionResult(
            selector_name=f"filter-{self.method}",
            num_features_selected=len(indices),
            num_features_total=len(self._scores),
            selected_indices=indices.tolist(),
            scores=[float(self._scores[i]) for i in indices],
            feature_names=(
                [self._feature_names[i] for i in indices]
                if self._feature_names
                else None
            ),
            metadata={
                "method": self.method,
                "k": self.k,
                "threshold": self.threshold,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_indices(self) -> NDArray[np.int64]:
        """Select feature indices based on scores."""
        if self._scores is None:
            raise RuntimeError("No scores computed")
        order = np.argsort(-self._scores)
        if self.k is not None and self.k > 0:
            order = order[: min(self.k, len(order))]
        if self.threshold > 0:
            order = order[self._scores[order] >= self.threshold]
        return np.sort(order)


# ------------------------------------------------------------------
# Filter implementations
# ------------------------------------------------------------------


def _filter_correlation(
    X: NDArray[np.float64],
    y: NDArray[np.float64] | NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """Pearson correlation between each feature and target.

    Args:
        X: Feature matrix.
        y: Target vector. If None, target is the first column of X.

    Returns:
        Absolute correlation scores per feature.
    """
    n_features = X.shape[1]
    if y is None:
        return np.ones(n_features, dtype=np.float64)

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    scores = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        col = X[:, j]
        if np.std(col) < 1e-12 or np.std(y_arr) < 1e-12:
            scores[j] = 0.0
        else:
            corr = np.corrcoef(col, y_arr)[0, 1]
            scores[j] = abs(corr) if not np.isnan(corr) else 0.0
    return scores


def _filter_mutual_info(
    X: NDArray[np.float64],
    y: NDArray[np.float64] | NDArray[np.int64] | None = None,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Mutual information between each feature and target.

    Uses a simple histogram-based estimator.

    Args:
        X: Feature matrix.
        y: Target vector.
        random_state: Random seed.

    Returns:
        MI scores per feature, normalized to [0, 1].
    """
    if y is None:
        n_features = X.shape[1]
        return np.ones(n_features, dtype=np.float64)

    try:
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    except ImportError:
        logger.warning("sklearn not installed; falling back to correlation")
        return _filter_correlation(X, y)

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_unique = len(np.unique(y_arr))

    if n_unique <= 10 and n_unique < len(y_arr) * 0.05:
        # Classification-like
        y_int = np.asarray(y_arr, dtype=np.int64)
        scores = mutual_info_classif(
            X, y_int, random_state=random_state, discrete_features="auto"
        )
    else:
        scores = mutual_info_regression(
            X, y_arr, random_state=random_state
        )

    # Handle possible NaN
    scores = np.nan_to_num(scores, nan=0.0)

    # Normalize to [0, 1]
    s_max = scores.max()
    if s_max > 0:
        scores = scores / s_max

    return scores


def _filter_chi2(
    X: NDArray[np.float64],
    y: NDArray[np.float64] | NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """Chi-squared statistic between each feature and discrete target.

    Args:
        X: Feature matrix (non-negative values expected).
        y: Discrete target vector.

    Returns:
        Chi-squared scores per feature.
    """
    if y is None:
        n_features = X.shape[1]
        return np.ones(n_features, dtype=np.float64)

    # Shift X to be non-negative
    X_nonneg = X - X.min(axis=0, keepdims=True) + 1e-8

    try:
        from sklearn.feature_selection import chi2

        scores, _ = chi2(X_nonneg, y)
        # Normalize
        s_max = scores.max()
        if s_max > 0:
            scores = scores / s_max
        return np.nan_to_num(scores, nan=0.0)
    except ImportError:
        logger.warning("sklearn not installed; falling back to correlation")
        return _filter_correlation(X, y)


def _filter_variance(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Variance-based scoring (higher variance = more informative).

    Args:
        X: Feature matrix.

    Returns:
        Variance per feature, normalized to [0, 1].
    """
    var = np.var(X, axis=0)
    v_max = var.max()
    if v_max > 0:
        var = var / v_max
    return np.nan_to_num(var, nan=0.0)


def _filter_anova(
    X: NDArray[np.float64],
    y: NDArray[np.float64] | NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """ANOVA F-value between each feature and categorical target.

    Args:
        X: Feature matrix.
        y: Categorical target.

    Returns:
        F-scores per feature, normalized.
    """
    if y is None:
        n_features = X.shape[1]
        return np.ones(n_features, dtype=np.float64)

    try:
        from sklearn.feature_selection import f_classif, f_regression
    except ImportError:
        logger.warning("sklearn not installed; falling back to correlation")
        return _filter_correlation(X, y)

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_unique = len(np.unique(y_arr))

    if n_unique <= 20:
        scores, _ = f_classif(X, y_arr.astype(np.int64))
    else:
        scores, _ = f_regression(X, y_arr)

    scores = np.nan_to_num(scores, nan=0.0)
    s_max = scores.max()
    if s_max > 0:
        scores = scores / s_max
    return scores