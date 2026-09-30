"""Deep Feature Synthesis (DFS) and automated feature selection for AstroML.

Enables automated relational feature generation, temporal aggregation,
feature selection (variance, correlation, missingness), and importance ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import mutual_info_classif

try:
    import featuretools as ft

    _HAS_FEATURETOOLS = True
except ImportError:
    _HAS_FEATURETOOLS = False
    ft = None

logger = logging.getLogger(__name__)

DEFAULT_AGG_PRIMITIVES = ["mean", "sum", "count", "std", "max", "min", "skew", "num_unique", "mode"]
DEFAULT_TRANS_PRIMITIVES = ["day", "month", "hour", "percentile", "cum_sum", "diff"]


class DeepFeatureSynthesizer:
    """Automated Deep Feature Synthesis generator using Featuretools.

    Parameters
    ----------
    target_dataframe_name : str
        Target entity to generate features for (e.g. 'accounts').
    agg_primitives : list[str] | None
        List of aggregation primitive names.
    trans_primitives : list[str] | None
        List of transformation primitive names.
    max_depth : int
        Maximum depth of feature stacking (1, 2, or 3). Default is 2.
    cutoff_time : pd.DataFrame | Any | None
        Cutoff time dataframe or timestamp for temporal calculations.
    training_window : str | None
        Window size before cutoff time to include in aggregations.
    max_features : int
        Maximum number of features to generate (-1 for all).
    verbose : bool
        Whether to print progress.
    """

    def __init__(
        self,
        target_dataframe_name: str = "accounts",
        agg_primitives: list[str] | None = None,
        trans_primitives: list[str] | None = None,
        max_depth: int = 2,
        cutoff_time: Any | None = None,
        training_window: str | None = None,
        max_features: int = -1,
        verbose: bool = False,
    ) -> None:
        if not _HAS_FEATURETOOLS or ft is None:
            raise ImportError(
                "Featuretools is required for DeepFeatureSynthesizer. "
                "Install it via: pip install featuretools"
            )
        self.target_dataframe_name = target_dataframe_name
        self.agg_primitives = agg_primitives or DEFAULT_AGG_PRIMITIVES
        self.trans_primitives = trans_primitives or DEFAULT_TRANS_PRIMITIVES
        self.max_depth = max_depth
        self.cutoff_time = cutoff_time
        self.training_window = training_window
        self.max_features = max_features
        self.verbose = verbose
        self.feature_defs_: list[Any] | None = None

    def fit_transform(self, entityset: ft.EntitySet) -> tuple[pd.DataFrame, list[Any]]:
        """Run Deep Feature Synthesis on an EntitySet.

        Parameters
        ----------
        entityset : ft.EntitySet
            Relational EntitySet.

        Returns
        -------
        tuple[pd.DataFrame, list[Any]]
            Calculated feature matrix and list of feature definitions.
        """
        # Filter primitives to those supported by current featuretools version
        valid_aggs = [
            p for p in self.agg_primitives if p in ft.primitives.get_aggregation_primitives()
        ]
        valid_trans = [
            p for p in self.trans_primitives if p in ft.primitives.get_transform_primitives()
        ]

        feature_matrix, feature_defs = ft.dfs(
            entityset=entityset,
            target_dataframe_name=self.target_dataframe_name,
            agg_primitives=valid_aggs if valid_aggs else None,
            trans_primitives=valid_trans if valid_trans else None,
            max_depth=self.max_depth,
            cutoff_time=self.cutoff_time,
            training_window=self.training_window,
            max_features=self.max_features,
            verbose=self.verbose,
        )

        # Convert categorical and boolean columns to numeric where appropriate
        feature_matrix = feature_matrix.copy()
        for col in feature_matrix.columns:
            if feature_matrix[col].dtype == "bool":
                feature_matrix[col] = feature_matrix[col].astype(float)
            elif pd.api.types.is_numeric_dtype(feature_matrix[col]):
                feature_matrix[col] = feature_matrix[col].fillna(0.0)

        self.feature_defs_ = feature_defs
        return feature_matrix, feature_defs

    def transform(
        self, entityset: ft.EntitySet, feature_defs: list[Any] | None = None
    ) -> pd.DataFrame:
        """Compute previously fitted feature definitions on a new EntitySet."""
        defs = feature_defs or self.feature_defs_
        if defs is None:
            raise ValueError("No feature definitions provided or fitted yet.")

        feature_matrix = ft.calculate_feature_matrix(
            features=defs,
            entityset=entityset,
            cutoff_time=self.cutoff_time,
            training_window=self.training_window,
            verbose=self.verbose,
        )
        return feature_matrix

    def save_feature_definitions(self, filepath: str) -> None:
        """Save fitted feature definitions to a JSON file."""
        if self.feature_defs_ is None:
            raise ValueError("No feature definitions to save.")
        ft.save_features(self.feature_defs_, filepath)

    @classmethod
    def load_feature_definitions(cls, filepath: str) -> list[Any]:
        """Load feature definitions from a JSON file."""
        if not _HAS_FEATURETOOLS or ft is None:
            raise ImportError("Featuretools is required to load feature definitions.")
        return ft.load_features(filepath)


# ---------------------------------------------------------------------------
# Feature Selection and Pruning
# ---------------------------------------------------------------------------


def prune_features(
    feature_matrix: pd.DataFrame,
    features: list[Any] | None = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    max_missing_rate: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """Prune uninformative, missing, or highly collinear features.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Input feature matrix.
    features : list[Any] | None
        Optional list of feature definitions corresponding to columns.
    variance_threshold : float
        Minimum variance required to retain a feature.
    correlation_threshold : float
        Maximum allowed Pearson correlation between two features.
    max_missing_rate : float
        Maximum fraction of missing values allowed.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Pruned DataFrame and list of retained feature column names.
    """
    df = feature_matrix.copy()

    # 1. Remove columns with excessive missing values
    missing_rates = df.isnull().mean()
    valid_cols = missing_rates[missing_rates <= max_missing_rate].index.tolist()
    df = df[valid_cols]

    # Impute remaining missing numeric values
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median() if not df[c].dropna().empty else 0.0)

    # 2. Select numeric columns for variance and correlation filtering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    retained_numeric: list[str] = []

    # Variance filter
    for c in numeric_cols:
        var = float(df[c].var()) if len(df[c]) > 1 else 0.0
        if var >= variance_threshold and not np.isnan(var):
            retained_numeric.append(c)

    # Correlation filter (drop one of each pair with correlation > threshold)
    to_drop: set[str] = set()
    if len(retained_numeric) > 1:
        corr_matrix = df[retained_numeric].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper_tri.columns:
            if any(upper_tri[col] > correlation_threshold):
                to_drop.add(col)

    final_numeric = [c for c in retained_numeric if c not in to_drop]
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    final_cols = final_numeric + non_numeric
    pruned_df = df[final_cols]

    return pruned_df, final_cols


# ---------------------------------------------------------------------------
# Feature Importance Ranking
# ---------------------------------------------------------------------------


def rank_feature_importance(
    feature_matrix: pd.DataFrame,
    target: pd.Series | np.ndarray,
    method: str = "random_forest",
    top_k: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank features by their predictive importance relative to a target variable.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Matrix of engineered features.
    target : pd.Series | np.ndarray
        Ground truth target (e.g. fraud label).
    method : str
        Method for calculating importance ('random_forest', 'gradient_boosting',
        'mutual_info', 'correlation').
    top_k : int | None
        Number of top features to return.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'importance', 'cumulative_importance', 'rank'].
    """
    df = feature_matrix.select_dtypes(include=[np.number]).copy()
    y = np.asarray(target)

    # Impute NaNs
    df = df.fillna(0.0)
    features = list(df.columns)

    if not features or len(y) == 0:
        return pd.DataFrame(columns=["feature", "importance", "cumulative_importance", "rank"])

    method_lower = method.lower()
    if method_lower == "random_forest":
        clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state)
        clf.fit(df, y)
        importances = clf.feature_importances_
    elif method_lower == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=random_state)
        clf.fit(df, y)
        importances = clf.feature_importances_
    elif method_lower == "mutual_info":
        importances = mutual_info_classif(df, y, random_state=random_state)
    elif method_lower == "correlation":
        corrs = [abs(float(df[c].corr(pd.Series(y)))) for c in df.columns]
        importances = np.nan_to_num(np.array(corrs), nan=0.0)
    else:
        raise ValueError(f"Unknown feature importance method: {method}")

    ranking_df = (
        pd.DataFrame(
            {
                "feature": features,
                "importance": importances,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    total_imp = ranking_df["importance"].sum()
    if total_imp > 0:
        ranking_df["cumulative_importance"] = (ranking_df["importance"] / total_imp).cumsum()
    else:
        ranking_df["cumulative_importance"] = 0.0

    ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)

    if top_k is not None:
        ranking_df = ranking_df.head(top_k)

    return ranking_df


# ---------------------------------------------------------------------------
# End-to-End Pipeline
# ---------------------------------------------------------------------------


class DFSPipeline:
    """Unified automated feature engineering pipeline."""

    def __init__(
        self,
        target_dataframe_name: str = "accounts",
        max_depth: int = 2,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        importance_method: str = "random_forest",
        top_k_features: int | None = 50,
    ) -> None:
        self.synthesizer = DeepFeatureSynthesizer(
            target_dataframe_name=target_dataframe_name,
            max_depth=max_depth,
        )
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.importance_method = importance_method
        self.top_k_features = top_k_features

        self.selected_features_: list[str] = []
        self.importance_ranking_: pd.DataFrame | None = None

    def fit_transform(
        self,
        entityset: ft.EntitySet,
        target: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Run full pipeline: Synthesis -> Prune -> Importance Ranking."""
        feature_matrix, _ = self.synthesizer.fit_transform(entityset)

        pruned_matrix, selected_cols = prune_features(
            feature_matrix=feature_matrix,
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
        )

        if target is not None:
            self.importance_ranking_ = rank_feature_importance(
                feature_matrix=pruned_matrix,
                target=target,
                method=self.importance_method,
                top_k=self.top_k_features,
            )
            if self.top_k_features is not None and not self.importance_ranking_.empty:
                top_cols = self.importance_ranking_["feature"].tolist()
                pruned_matrix = pruned_matrix[top_cols]
                selected_cols = top_cols

        self.selected_features_ = selected_cols
        return pruned_matrix
