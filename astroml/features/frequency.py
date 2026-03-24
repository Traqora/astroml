"""Compute transaction frequency metrics for blockchain accounts.

This module contains helpers used to build frequency-based features from
transaction data, including daily activity counts and burstiness metrics.
Inputs are pandas DataFrames with configurable timestamp and account columns.
"""
from typing import Hashable, Union

import numpy as np
import pandas as pd

Number = Union[float, int]
ArrayLike = Union[Number, np.ndarray, pd.Series, list, tuple]


def _validate_dataframe(
    df: pd.DataFrame,
    timestamp_col: str,
    account_col: str,
) -> None:
    """Validate required columns and normalize the timestamp column.

    Args:
        df: DataFrame to validate.
        timestamp_col: Expected timestamp column name.
        account_col: Expected account column name.

    Raises:
        ValueError: If required columns are missing, nulls are present,
            or timestamps cannot be interpreted as datetimes.
    """
    required_cols = [timestamp_col, account_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        missing = ", ".join(f"'{col}'" for col in missing_cols)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if df[timestamp_col].isna().any():
        raise ValueError(f"Column '{timestamp_col}' contains null values")

    if df[account_col].isna().any():
        raise ValueError(f"Column '{account_col}' contains null values")

    if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        return

    try:
        if pd.api.types.is_numeric_dtype(df[timestamp_col]):
            numeric_timestamps = pd.to_numeric(df[timestamp_col], errors="raise")
            max_abs_value = numeric_timestamps.abs().max()

            if max_abs_value < 1e11:
                unit = "s"
            elif max_abs_value < 1e14:
                unit = "ms"
            elif max_abs_value < 1e17:
                unit = "us"
            else:
                unit = "ns"

            converted = pd.to_datetime(
                numeric_timestamps,
                unit=unit,
                errors="raise",
            )
        else:
            converted = pd.to_datetime(df[timestamp_col], errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Column '{timestamp_col}' must contain datetime values or parseable timestamps"
        ) from exc

    df[timestamp_col] = converted


def _extract_daily_counts(
    timestamps: pd.Series,
) -> np.ndarray:
    """Convert timestamps to array of daily transaction counts.

    This function processes a series of timestamps and returns an array of
    transaction counts per day, covering the full time window from the first
    to the last transaction. Days with no transactions are included as zeros.

    Args:
        timestamps: Series of datetime objects representing transaction times.

    Returns:
        Array of daily transaction counts covering the full time window.
        Returns empty array if timestamps is empty.

    Notes:
        - Time window spans from first to last transaction date (inclusive)
        - Days with zero transactions are explicitly included as 0 counts
        - Timestamps are converted to date resolution (day granularity)
        - Order of input timestamps does not affect the result

    Examples:
        >>> import pandas as pd
        >>> timestamps = pd.Series(pd.to_datetime([
        ...     '2024-01-01', '2024-01-01', '2024-01-03'
        ... ]))
        >>> counts = _extract_daily_counts(timestamps)
        >>> counts.tolist()
        [2, 0, 1]
    """
    if len(timestamps) == 0:
        return np.array([])

    dates = timestamps.dt.date

    if len(timestamps) == 1:
        return np.array([1])

    min_date = dates.min()
    max_date = dates.max()
    date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    daily_counts = dates.value_counts()
    daily_counts = daily_counts.reindex(date_range.date, fill_value=0)
    return daily_counts.values


def _compute_burstiness(mean: float, std: float) -> float:
    """Calculate burstiness metric from mean and standard deviation.

    The burstiness metric quantifies temporal clustering of transactions using
    the formula B = (std - mean) / (std + mean). The result is bounded in
    [-1, 1] with intuitive interpretation:

    - B ~= 1: Highly bursty (high variance, clustered transactions)
    - B ~= 0: Random/Poisson-like (variance equals mean)
    - B ~= -1: Highly regular (low variance, periodic transactions)

    Args:
        mean: Mean of daily transaction counts (mean >= 0).
        std: Standard deviation of daily counts (std >= 0).

    Returns:
        Burstiness value in [-1, 1]. Returns 0.0 when both mean and std are 0.

    Notes:
        - When std + mean = 0 (both zero), returns 0.0 by definition
        - When std = 0 (perfectly regular), returns -1.0
        - When std >> mean (highly variable), approaches 1.0
        - Result is automatically bounded in [-1, 1] by the formula

    Examples:
        >>> _compute_burstiness(5.0, 2.0)
        -0.42857142857142855
        >>> _compute_burstiness(0.0, 0.0)
        0.0
        >>> _compute_burstiness(5.0, 0.0)
        -1.0
    """
    if mean + std == 0.0:
        return 0.0

    return (std - mean) / (std + mean)


def _compute_frequency_metrics_for_timestamps(
    timestamps: pd.Series,
) -> dict[str, float]:
    """Compute frequency metrics for one account's validated timestamp series.

    Args:
        timestamps: Validated timestamp series for a single account.

    Returns:
        Dictionary containing the mean daily transaction count, the sample
        standard deviation of daily transaction counts, and burstiness.
    """
    if len(timestamps) == 0:
        return {"mean_tx_per_day": 0.0, "std_tx_per_day": 0.0, "burstiness": 0.0}

    daily_counts = _extract_daily_counts(timestamps)
    mean = float(np.mean(daily_counts))
    std = float(np.std(daily_counts, ddof=1)) if len(daily_counts) > 1 else 0.0
    burstiness = _compute_burstiness(mean, std)

    return {
        "mean_tx_per_day": mean,
        "std_tx_per_day": std,
        "burstiness": burstiness,
    }


def compute_frequency_metrics(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    account_col: str = "account",
) -> pd.DataFrame:
    """Compute frequency metrics for each account in a transaction DataFrame.

    Args:
        df: Transaction DataFrame containing account and timestamp columns.
        timestamp_col: Name of the timestamp column.
        account_col: Name of the account identifier column.

    Returns:
        DataFrame with one row per account and these columns:
        ``account_col``, ``mean_tx_per_day``, ``std_tx_per_day``, and
        ``burstiness``.

    Notes:
        Validation and timestamp normalization are delegated to
        :func:`_validate_dataframe`. Metric formulas are delegated to
        :func:`_compute_frequency_metrics_for_timestamps` so the batch and
        single-account paths stay consistent.
    """
    working_df = df.copy()
    _validate_dataframe(working_df, timestamp_col=timestamp_col, account_col=account_col)

    metric_rows = []
    for account_value, account_df in working_df.groupby(account_col, sort=False):
        metric_rows.append(
            {
                account_col: account_value,
                **_compute_frequency_metrics_for_timestamps(account_df[timestamp_col]),
            }
        )

    return pd.DataFrame(metric_rows)


def compute_account_frequency(
    df: pd.DataFrame,
    account_id: Hashable,
    timestamp_col: str = "timestamp",
    account_col: str = "account",
) -> dict[str, float]:
    """Compute transaction-frequency metrics for one specified account.

    Args:
        df: Transaction DataFrame containing at least the account and timestamp
            columns expected by the batch computation path.
        account_id: Account identifier whose frequency metrics should be
            returned.
        timestamp_col: Name of the timestamp column. Defaults to
            ``"timestamp"``.
        account_col: Name of the account identifier column. Defaults to
            ``"account"``.

    Returns:
        Dictionary with exactly these keys:
        ``"mean_tx_per_day"``, ``"std_tx_per_day"``, and ``"burstiness"``.

    Raises:
        ValueError: If the requested account does not exist in ``account_col``
            or if the DataFrame fails batch validation.

    Notes:
        This is a thin wrapper around :func:`compute_frequency_metrics`. It
        validates the DataFrame using the same path as the batch function,
        filters to the requested account, and extracts that account's row from
        the batch result. Single-day behavior, custom column handling, and any
        edge-case ``NaN`` values are therefore inherited directly from the
        batch implementation.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "account": ["acct-1", "acct-1", "acct-2"],
        ...     "timestamp": ["2024-01-01", "2024-01-03", "2024-01-02"],
        ... })
        >>> compute_account_frequency(df, "acct-1")
        {'mean_tx_per_day': 0.6666666666666666, 'std_tx_per_day': 0.5773502691896258, 'burstiness': -0.07179676972449088}

        Custom column names are supported when they match the batch API:

        >>> renamed = df.rename(columns={"account": "acct", "timestamp": "ts"})
        >>> compute_account_frequency(renamed, "acct-2", account_col="acct", timestamp_col="ts")
        {'mean_tx_per_day': 1.0, 'std_tx_per_day': 0.0, 'burstiness': -1.0}
    """
    working_df = df.copy()
    _validate_dataframe(working_df, timestamp_col=timestamp_col, account_col=account_col)

    account_df = working_df.loc[working_df[account_col] == account_id]
    if account_df.empty:
        raise ValueError(f"Account {account_id!r} not found in column '{account_col}'")

    batch_metrics = compute_frequency_metrics(
        account_df,
        timestamp_col=timestamp_col,
        account_col=account_col,
    )
    metric_row = batch_metrics.iloc[0]

    return {
        "mean_tx_per_day": float(metric_row["mean_tx_per_day"]),
        "std_tx_per_day": float(metric_row["std_tx_per_day"]),
        "burstiness": float(metric_row["burstiness"]),
    }
