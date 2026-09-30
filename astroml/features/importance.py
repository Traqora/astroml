"""Feature importance and selection utilities.

Issue #744: provides permutation-importance style analysis and a simple
threshold/score-based selector that works with the feature store output
(pandas DataFrames whose columns are features).

Design notes:
    - ``permutation_importance`` is model-agnostic: it measures the drop in a
      scorer when a single feature's column is randomly shuffled. It needs no
      access to model internals, so it works with any fitted predictor that
      implements ``predict`` (or any ``(X) -> score`` callable).
    - ``select_features`` applies simple, transparent filters (variance,
      missing rate, correlation) and/or an importance threshold / top-k cut,
      mirroring the knobs documented in ``configs/feature_engineering.yaml``.
    - Results are plain dataclasses so they can be logged, cached (#743), or
      attached to an audit trail (#757) without extra dependencies beyond
      pandas/numpy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureImportance",
    "FeatureImportanceReport",
    "SelectionReport",
    "compute_permutation_importance",
    "default_selection_filters",
    "select_features",
]


# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureImportance:
    """Importance score for a single feature."""

    feature: str
    importance: float
    #: Mean scorer value observed while this feature was permuted.
    permuted_score_mean: float
    #: Std of the scorer value across permutation repeats.
    permuted_score_std: float


@dataclass(frozen=True)
class FeatureImportanceReport:
    """Permutation-importance report for a set of features."""

    baseline_score: float
    importances: tuple[FeatureImportance, ...]
    scorer_name: str = "score"
    n_repeats: int = 5
    random_state: int | None = None

    def ranked(self, descending: bool = True) -> list[FeatureImportance]:
        """Importances sorted by score."""
        return sorted(self.importances, key=lambda i: i.importance, reverse=descending)

    def top_k(self, k: int) -> list[FeatureImportance]:
        """The ``k`` most important features."""
        return self.ranked()[: max(k, 0)]

    def to_frame(self) -> pd.DataFrame:
        """Render as a DataFrame sorted by descending importance."""
        rows = [
            {
                "feature": imp.feature,
                "importance": imp.importance,
                "permuted_score_mean": imp.permuted_score_mean,
                "permuted_score_std": imp.permuted_score_std,
            }
            for imp in self.ranked()
        ]
        return pd.DataFrame(
            rows, columns=["feature", "importance", "permuted_score_mean", "permuted_score_std"]
        )


def compute_permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    features: Sequence[str] | None = None,
    *,
    scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_repeats: int = 5,
    random_state: int | None = 42,
) -> FeatureImportanceReport:
    """Permutation-importance analysis over a fitted predictor.

    For each feature, the column is randomly shuffled ``n_repeats`` times and
    the drop in ``scorer(y, predict(X_permuted))`` relative to the unshuffled
    baseline is recorded. A large drop means the model relied on that feature.

    Args:
        predict: Callable mapping a feature DataFrame to predictions.
        X: Feature matrix (one column per feature, one row per sample).
        y: Ground-truth targets aligned with ``X``.
        features: Columns to analyze; defaults to all columns of ``X``.
        scorer: ``scorer(y_true, y_pred) -> float``; defaults to R2 for
            regression-shaped targets and accuracy when targets look binary.
        n_repeats: Shuffles per feature.
        random_state: Seed for reproducibility.

    Returns:
        A :class:`FeatureImportanceReport` with per-feature importances.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    columns = list(features) if features is not None else list(X.columns)
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise KeyError(f"features not present in X: {missing}")

    y_arr = np.asarray(y)
    scorer_fn = scorer or _default_scorer(y_arr)
    rng = np.random.default_rng(random_state)

    baseline = float(scorer_fn(y_arr, np.asarray(predict(X))))

    importances: list[FeatureImportance] = []
    for column in columns:
        scores: list[float] = []
        original = X[column].to_numpy(copy=True)
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(original)
            scores.append(float(scorer_fn(y_arr, np.asarray(predict(shuffled)))))
        drop = baseline - float(np.mean(scores))
        importances.append(
            FeatureImportance(
                feature=column,
                importance=drop,
                permuted_score_mean=float(np.mean(scores)),
                permuted_score_std=float(np.std(scores)),
            )
        )

    return FeatureImportanceReport(
        baseline_score=baseline,
        importances=tuple(importances),
        scorer_name=getattr(scorer_fn, "__name__", "score"),
        n_repeats=n_repeats,
        random_state=random_state,
    )


def _default_scorer(y: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
    """R2 for continuous targets, accuracy for binary ones."""
    unique = np.unique(y[np.isfinite(y)]) if np.issubdtype(y.dtype, np.number) else np.unique(y)
    if len(unique) == 2:
        return _accuracy
    return _r2_score


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator == 0:
        return 0.0
    numerator = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - numerator / denominator


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass
class SelectionReport:
    """Result of running the feature selector on a feature matrix.

    Attributes:
        selected: Columns that survived the filters/thresholds.
        dropped: Mapping of dropped column to the reason (human-readable).
        importances: Optional per-feature importance used for the cut.
    """

    selected: list[str]
    dropped: dict[str, str] = field(default_factory=dict)
    importances: dict[str, float] = field(default_factory=dict)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` reduced to the selected columns."""
        present = [c for c in self.selected if c in X.columns]
        return X[present]


def default_selection_filters(
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    max_missing_rate: float = 0.3,
) -> list[Callable[[pd.DataFrame], dict[str, str]]]:
    """Build the default filter stack used by ``select_features``.

    Mirrors the thresholds documented in ``configs/feature_engineering.yaml``:
    near-constant features, collinear features, and features with too many
    missing values are dropped.
    """

    def variance_filter(X: pd.DataFrame) -> dict[str, str]:
        reasons: dict[str, str] = {}
        variances = X.var(numeric_only=True)
        for column, variance in variances.items():
            if float(variance) < variance_threshold:
                reasons[column] = f"variance {variance:.6g} < {variance_threshold}"
        return reasons

    def missing_filter(X: pd.DataFrame) -> dict[str, str]:
        reasons: dict[str, str] = {}
        rates = X.isna().mean()
        for column, rate in rates.items():
            if float(rate) > max_missing_rate:
                reasons[column] = f"missing rate {rate:.2%} > {max_missing_rate:.0%}"
        return reasons

    def correlation_filter(X: pd.DataFrame) -> dict[str, str]:
        reasons: dict[str, str] = {}
        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return reasons
        corr = numeric.corr(numeric_only=True).abs()
        # Guard against constant (all-NaN after abs) correlation matrices,
        # e.g. pandas 3.0 raises on idxmax over an all-NaN column.
        if not corr.notna().to_numpy().any():
            return reasons
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for column in upper.columns:
            series = upper[column].dropna()
            if series.empty:
                continue
            partner = series.idxmax()
            value = series.max()
            if pd.notna(value) and float(value) > correlation_threshold:
                reasons[column] = (
                    f"correlation {float(value):.3f} with {partner} > {correlation_threshold}"
                )
        return reasons

    return [variance_filter, missing_filter, correlation_filter]


def select_features(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series | None = None,
    *,
    predict: Callable[[pd.DataFrame], np.ndarray] | None = None,
    filters: Sequence[Callable[[pd.DataFrame], dict[str, str]]] | None = None,
    importance_threshold: float | None = None,
    top_k: int | None = None,
    n_repeats: int = 5,
    random_state: int | None = 42,
) -> SelectionReport:
    """Select features using transparent filters and/or importance scores.

    Order of operations:
        1. Static filters (variance / missing / correlation by default) drop
           obviously unusable columns.
        2. If ``predict`` (and ``y``) are supplied, permutation importance is
           computed on the survivors.
        3. ``importance_threshold`` and ``top_k`` cut the survivors by
           importance. When importance was not computed, those two options
           are ignored (they need scores to work).

    Args:
        X: Feature matrix (rows = samples, columns = features).
        y: Targets, required only when ``predict`` is given.
        predict: Fitted predictor callable for importance computation.
        filters: Custom filter stack; defaults to
            :func:`default_selection_filters`.
        importance_threshold: Keep features whose importance is at or above
            this value (only with ``predict``).
        top_k: Keep the ``k`` most important features (only with ``predict``).
        n_repeats: Permutation repeats per feature.
        random_state: Seed for the permutation RNG.

    Returns:
        A :class:`SelectionReport` with the selected columns and drop reasons.
    """
    if X.empty:
        return SelectionReport(selected=[])

    remaining = list(X.columns)
    dropped: dict[str, str] = {}

    filter_stack = list(filters) if filters is not None else default_selection_filters()
    for filter_fn in filter_stack:
        reasons = filter_fn(X[remaining])
        for column, reason in reasons.items():
            if column in remaining:
                remaining.remove(column)
                dropped[column] = reason
                logger.debug("dropped %s: %s", column, reason)

    importances: dict[str, float] = {}
    if predict is not None:
        if y is None:
            raise ValueError("y is required when predict is provided")
        report = compute_permutation_importance(
            predict,
            X[remaining],
            y,
            features=remaining,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        importances = {imp.feature: imp.importance for imp in report.importances}

        if importance_threshold is not None:
            for column in list(remaining):
                if importances.get(column, 0.0) < importance_threshold:
                    remaining.remove(column)
                    dropped[column] = (
                        f"importance {importances.get(column, 0.0):.6g} < {importance_threshold}"
                    )

        if top_k is not None and len(remaining) > top_k:
            ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
            keep = {column for column, _ in ranked[:top_k]}
            for column in list(remaining):
                if column not in keep:
                    remaining.remove(column)
                    dropped[column] = f"outside top_{top_k} by importance"

    return SelectionReport(selected=remaining, dropped=dropped, importances=importances)
