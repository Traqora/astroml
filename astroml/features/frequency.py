"""Compute transaction frequency metrics for blockchain accounts.

This module provides utilities for analyzing temporal transaction patterns by
computing frequency-based metrics for blockchain accounts. The primary metrics
include:

- **Mean transactions per day**: Average daily transaction rate over an
  account's active period
- **Standard deviation**: Variability in daily transaction counts
- **Burstiness metric**: Normalized measure of temporal clustering, defined as
  (σ - μ) / (σ + μ), bounded in [-1, 1]

The burstiness metric provides intuitive interpretation:
- B ≈ 1: Highly bursty (transactions clustered in time)
- B ≈ 0: Random/Poisson-like (memoryless process)
- B ≈ -1: Highly regular (periodic transactions)

Inputs are pandas DataFrames with configurable column names for timestamps and
account identifiers. The module handles edge cases gracefully, including empty
data, single-day transactions, and various timestamp formats.
"""
from typing import Union, Dict

import numpy as np
import pandas as pd

Number = Union[float, int]
ArrayLike = Union[Number, np.ndarray, pd.Series, list, tuple]



def _validate_dataframe(
    df: pd.DataFrame,
    timestamp_col: str,
    account_col: str,
) -> None:
    """Validate input DataFrame structure and content.
    
    Args:
        df: DataFrame to validate.
        timestamp_col: Expected timestamp column name.
        account_col: Expected account column name.
        
    Raises:
        ValueError: If validation fails with a descriptive message.
        
    Notes:
        - Checks that required columns exist in the DataFrame
        - Verifies no null values in timestamp or account columns
        - Validates timestamp column is datetime or numeric (Unix timestamp)
        - Converts numeric timestamps to datetime if needed
    """
    # Check required columns exist
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")
    if account_col not in df.columns:
        raise ValueError(f"Column '{account_col}' not found in DataFrame")
    
    # Check for null values
    if df[timestamp_col].isnull().any():
        raise ValueError(f"Column '{timestamp_col}' contains null values")
    if df[account_col].isnull().any():
        raise ValueError(f"Column '{account_col}' contains null values")
    
    # Validate timestamp type
    if not (pd.api.types.is_datetime64_any_dtype(df[timestamp_col]) or 
            pd.api.types.is_numeric_dtype(df[timestamp_col])):
        raise ValueError(
            f"Column '{timestamp_col}' must be datetime or numeric (Unix timestamp)"
        )


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
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')

    # Count transactions per day using value_counts
    daily_counts = dates.value_counts()

    # Fill missing days with 0
    daily_counts = daily_counts.reindex(date_range.date, fill_value=0)

    # Return as numpy array
    return daily_counts.values



def compute_frequency_metrics(
    df: pd.DataFrame,
    timestamp_col: str = "created_at",
    account_col: str = "account_id",
) -> pd.DataFrame:
    """Compute transaction frequency metrics for each account in a DataFrame.

    For each unique account, three temporal frequency metrics are computed
    over the account's active window (first to last transaction date,
    inclusive):

    - **mean_tx_per_day**: mean number of transactions per calendar day
    - **std_tx_per_day**: sample standard deviation (ddof=1) of daily counts
    - **burstiness**: normalised clustering metric ``(σ - μ) / (σ + μ)``

    Args:
        df: Transaction DataFrame. Must contain *timestamp_col* and
            *account_col*; extra columns are ignored.
        timestamp_col: Name of the column containing transaction timestamps.
            Accepted types: ``datetime64`` or numeric (Unix epoch seconds).
            Defaults to ``"created_at"``.
        account_col: Name of the column containing account identifiers.
            Values may be any hashable type (str, int, …).
            Defaults to ``"account_id"``.

    Returns:
        DataFrame with one row per unique account and columns:

        - ``account``          – account identifier (original type preserved)
        - ``mean_tx_per_day``  – mean daily transaction count (float)
        - ``std_tx_per_day``   – sample std of daily counts; 0.0 for
          single-day windows (float)
        - ``burstiness``       – burstiness in ``[-1, 1]`` (float)

        Returns an empty DataFrame with those columns when *df* is empty.

    Notes:
        - Uses ``ddof=1`` (sample standard deviation). Returns ``std=0.0``
          for accounts whose entire history falls within a single calendar
          day (only one data point, so sample std is undefined).
        - Numeric timestamps are treated as Unix epoch **seconds** and
          converted via ``pd.to_datetime(..., unit="s")``.
        - The original account identifier type (str, int, …) is preserved
          in the ``account`` output column.
        - Each row in *df* counts as one transaction; duplicate rows are
          each counted separately.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "account_id": ["alice", "alice", "bob"],
        ...     "created_at": pd.to_datetime(["2024-01-01", "2024-01-02",
        ...                                   "2024-01-01"]),
        ... })
        >>> compute_frequency_metrics(df)
          account  mean_tx_per_day  std_tx_per_day  burstiness
        0   alice              1.0             0.0        -1.0
        1     bob              1.0             0.0        -1.0
    """
    _validate_dataframe(df, timestamp_col, account_col)

    _SCHEMA = ["account", "mean_tx_per_day", "std_tx_per_day", "burstiness"]

    if df.empty:
        return pd.DataFrame(columns=_SCHEMA)

    # Convert Unix epoch seconds to datetime if needed
    working = df
    if pd.api.types.is_numeric_dtype(df[timestamp_col]):
        working = df.copy()
        working[timestamp_col] = pd.to_datetime(working[timestamp_col], unit="s")

    records: list[Dict] = []
    for account, group in working.groupby(account_col):
        daily_counts = _extract_daily_counts(group[timestamp_col])
        mean = float(np.mean(daily_counts))
        # ddof=1 requires at least 2 data points; single-day window → std=0.0
        std = float(np.std(daily_counts, ddof=1)) if len(daily_counts) > 1 else 0.0
        burstiness = _compute_burstiness(mean, std)
        records.append(
            {
                "account": account,
                "mean_tx_per_day": mean,
                "std_tx_per_day": std,
                "burstiness": burstiness,
            }
        )

    return pd.DataFrame(records)


def _compute_burstiness(mean: float, std: float) -> float:
    """Calculate burstiness metric from mean and standard deviation.

    The burstiness metric quantifies temporal clustering of transactions using
    the formula B = (σ - μ) / (σ + μ), where σ is standard deviation and μ is
    mean. The result is bounded in [-1, 1] with intuitive interpretation:
    
    - B ≈ 1: Highly bursty (high variance, clustered transactions)
    - B ≈ 0: Random/Poisson-like (variance equals mean)
    - B ≈ -1: Highly regular (low variance, periodic transactions)

    Args:
        mean: Mean of daily transaction counts (μ ≥ 0).
        std: Standard deviation of daily counts (σ ≥ 0).

    Returns:
        Burstiness value in [-1, 1]. Returns 0.0 when both mean and std are 0.

    Notes:
        - When σ + μ = 0 (both zero), returns 0.0 by definition
        - When σ = 0 (perfectly regular), returns -1.0
        - When σ >> μ (highly variable), approaches 1.0
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
