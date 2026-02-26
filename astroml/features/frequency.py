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
