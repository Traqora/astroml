"""Compute transaction frequency metrics for blockchain accounts.

This module contains helpers used to build frequency-based features from
transaction data, including daily activity counts and burstiness metrics.
Inputs are pandas DataFrames with configurable timestamp and account columns.
"""
from typing import Union

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

            # Infer UNIX timestamp unit by magnitude.
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

    df.loc[:, timestamp_col] = converted


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
    # Handle empty timestamps
    if len(timestamps) == 0:
        return np.array([])

    # Convert timestamps to dates (day resolution)
    dates = timestamps.dt.date

    # Handle single timestamp
    if len(timestamps) == 1:
        return np.array([1])

    # Determine first and last transaction dates
    min_date = dates.min()
    max_date = dates.max()

    # Create complete date range from first to last
    date_range = pd.date_range(start=min_date, end=max_date, freq="D")

    # Count transactions per day using value_counts
    daily_counts = dates.value_counts()

    # Fill missing days with 0
    daily_counts = daily_counts.reindex(date_range.date, fill_value=0)

    # Return as numpy array
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
    # Handle edge case: when mean + std == 0, return 0.0
    if mean + std == 0.0:
        return 0.0

    # Calculate burstiness: (std - mean) / (std + mean)
    return (std - mean) / (std + mean)


def compute_account_frequency(
    timestamps: pd.Series,
) -> Dict[str, float]:
    """Compute frequency metrics for a single account's transaction timestamps.

    A per-account convenience function that wraps the internal helpers to
    produce all three frequency metrics for one set of timestamps at a time.
    For batch processing of a full DataFrame with multiple accounts, see
    :func:`compute_frequency_metrics` (available after merging #47).

    Args:
        timestamps: Transaction timestamps for a single account. Accepts a
            ``datetime64`` Series or a numeric (Unix epoch seconds) Series.
            An empty Series is valid and returns all-zero metrics.

    Returns:
        Dictionary with three keys:

        - ``"mean_tx_per_day"``  – mean number of transactions per calendar
          day over the account's active window (float)
        - ``"std_tx_per_day"``   – sample standard deviation (ddof=1) of
          daily counts; 0.0 for a single-day window (float)
        - ``"burstiness"``       – normalised clustering metric in ``[-1, 1]``
          (float)

    Notes:
        - Uses ``ddof=1`` for standard deviation. Returns ``std=0.0`` for
          accounts whose entire history falls within a single calendar day
          (only one data point, so sample std is undefined).
        - Numeric timestamps are treated as Unix epoch **seconds** and
          converted via ``pd.to_datetime(..., unit="s")``.
        - An empty Series returns all-zero metrics by convention.

    Examples:
        >>> import pandas as pd
        >>> ts = pd.Series(pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-03']))
        >>> result = compute_account_frequency(ts)
        >>> result['mean_tx_per_day']
        1.0
        >>> result['std_tx_per_day']
        1.0
        >>> result['burstiness']
        0.0

        Empty timestamps return all-zero metrics:

        >>> compute_account_frequency(pd.Series([], dtype='datetime64[ns]'))
        {'mean_tx_per_day': 0.0, 'std_tx_per_day': 0.0, 'burstiness': 0.0}
    """
    if pd.api.types.is_numeric_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps, unit="s")

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
