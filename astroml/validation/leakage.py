"""Feature leakage detection and temporal split enforcement.

This module provides utilities to prevent future information from leaking into
training sets when working with time-series blockchain transaction data.

Two main capabilities:

1. **Temporal split enforcement** — split DataFrames by time so that training
   data strictly precedes test data, and validate that no overlap exists.

2. **Automated warning system** — scan DataFrames for common leakage
   indicators such as unsorted timestamps, constant features, or features
   with near-perfect correlation to the target variable.
"""
import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from typing import NamedTuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class LeakageError(Exception):
    """Raised when a hard temporal leakage violation is detected."""


class LeakageWarning(NamedTuple):
    """Structured description of a potential leakage issue.

    Attributes:
        column: Name of the column involved (or ``"_time"`` for sort issues).
        warning_type: Category tag — one of ``"unsorted"``, ``"constant"``,
            or ``"target_correlation"``.
        message: Human-readable explanation.
    """

    column: str
    warning_type: str
    message: str


# ---------------------------------------------------------------------------
# Temporal split enforcement
# ---------------------------------------------------------------------------

def temporal_train_test_split(
    df: pd.DataFrame,
    time_col: str,
    *,
    cutoff: Optional[Any] = None,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test sets respecting temporal order.

    Two modes of operation:

    * **Cutoff mode** (when *cutoff* is provided): rows where
      ``time_col < cutoff`` go to train, the rest to test.
    * **Ratio mode** (default): the DataFrame is sorted by *time_col* and
      split at ``int(len(df) * train_ratio)``.

    Args:
        df: Input data.
        time_col: Column containing timestamps or any sortable time
            representation.
        cutoff: Explicit temporal cutoff value.  When provided, *train_ratio*
            is ignored.
        train_ratio: Fraction of rows to assign to the training set when
            using ratio mode.  Must be in the open interval ``(0, 1)``.

    Returns:
        ``(train_df, test_df)`` tuple of DataFrames.

    Raises:
        ValueError: If *time_col* is missing, contains null values, or
            *train_ratio* is outside ``(0, 1)``.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")

    if df[time_col].isna().any():
        raise ValueError(
            f"Column '{time_col}' contains null values; "
            "remove or fill them before splitting"
        )

    if df.empty:
        return df.copy(), df.copy()

    if cutoff is not None:
        train_mask = df[time_col] < cutoff
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[~train_mask].copy()
    else:
        if not (0 < train_ratio < 1):
            raise ValueError(
                f"train_ratio must be in (0, 1), got {train_ratio}"
            )
        sorted_df = df.sort_values(time_col).reset_index(drop=True)
        split_idx = int(len(sorted_df) * train_ratio)
        train_df = sorted_df.iloc[:split_idx].copy()
        test_df = sorted_df.iloc[split_idx:].copy()

    if train_df.empty:
        warnings.warn(
            "Train partition is empty — cutoff may be before all data",
            UserWarning,
            stacklevel=2,
        )
    if test_df.empty:
        warnings.warn(
            "Test partition is empty — cutoff may be after all data",
            UserWarning,
            stacklevel=2,
        )

    return train_df, test_df


def validate_temporal_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str,
) -> bool:
    """Verify that a train/test split has no temporal overlap.

    Args:
        train_df: Training partition.
        test_df: Test partition.
        time_col: Column containing timestamps.

    Returns:
        ``True`` if the split is clean (no overlap).

    Raises:
        LeakageError: If the maximum training timestamp is not strictly less
            than the minimum test timestamp.
        ValueError: If *time_col* is missing from either DataFrame.
    """
    for label, partition in [("train", train_df), ("test", test_df)]:
        if time_col not in partition.columns:
            raise ValueError(
                f"Column '{time_col}' not found in {label} DataFrame"
            )

    # Empty partitions are trivially valid.
    if train_df.empty or test_df.empty:
        return True

    train_max = train_df[time_col].max()
    test_min = test_df[time_col].min()

    if train_max >= test_min:
        raise LeakageError(
            f"Temporal overlap detected: train max ({train_max}) "
            f">= test min ({test_min})"
        )

    return True


# ---------------------------------------------------------------------------
# Automated warning system
# ---------------------------------------------------------------------------

def check_feature_leakage(
    df: pd.DataFrame,
    time_col: str,
    feature_cols: Optional[list[str]] = None,
) -> list[LeakageWarning]:
    """Scan a DataFrame for common feature-leakage indicators.

    Runs two deterministic checks:

    1. **Temporal sort** — warns if *time_col* is not monotonically
       non-decreasing.
    2. **Constant columns** — flags numeric feature columns with zero
       variance, which often indicate a leakage artifact or a useless
       feature.

    Args:
        df: Input data.
        time_col: Column containing timestamps.
        feature_cols: Columns to inspect.  Defaults to all numeric columns
            except *time_col*.

    Returns:
        List of :class:`LeakageWarning` instances.  An empty list means no
        issues were detected.
    """
    results: list[LeakageWarning] = []

    if df.empty:
        return results

    # -- Temporal sort check --------------------------------------------------
    if time_col in df.columns and not df[time_col].is_monotonic_increasing:
        w = LeakageWarning(
            column=time_col,
            warning_type="unsorted",
            message=(
                f"Column '{time_col}' is not sorted in ascending order; "
                "data may not respect temporal boundaries"
            ),
        )
        results.append(w)
        warnings.warn(w.message, UserWarning, stacklevel=2)

    # -- Constant column check ------------------------------------------------
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != time_col]

    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=False) <= 1:
            w = LeakageWarning(
                column=col,
                warning_type="constant",
                message=(
                    f"Column '{col}' has zero variance (constant); "
                    "this may indicate a leakage artifact or a useless feature"
                ),
            )
            results.append(w)
            warnings.warn(w.message, UserWarning, stacklevel=2)

    return results


def check_target_leakage(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[list[str]] = None,
    threshold: float = 0.95,
) -> list[LeakageWarning]:
    """Check whether any feature has near-perfect correlation with the target.

    Computes Pearson correlation between each feature column and
    *target_col*.  Features exceeding *threshold* are flagged.

    .. note::

       This only detects *linear* relationships.  Non-linear leakage (e.g. a
       feature that is a monotone transform of the target) will not be caught.

    Args:
        df: Input data.
        target_col: Name of the target / label column.
        feature_cols: Columns to check.  Defaults to all numeric columns
            except *target_col*.
        threshold: Absolute correlation above which a feature is flagged.
            Must be in ``(0, 1]``.

    Returns:
        List of :class:`LeakageWarning` instances.  An empty list means no
        issues were detected.
    """
    results: list[LeakageWarning] = []

    if df.empty or target_col not in df.columns:
        return results

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]

    target = df[target_col]

    # Skip if the target itself has zero variance.
    if target.std() == 0:
        return results

    for col in feature_cols:
        if col not in df.columns:
            continue

        series = df[col]

        # Skip zero-variance features (correlation is undefined).
        if series.std() == 0:
            continue

        corr = target.corr(series)

        if abs(corr) > threshold:
            w = LeakageWarning(
                column=col,
                warning_type="target_correlation",
                message=(
                    f"Column '{col}' has high correlation ({corr:.4f}) "
                    f"with target '{target_col}'; possible target leakage"
                ),
            )
            results.append(w)
            warnings.warn(w.message, UserWarning, stacklevel=2)

    return results
