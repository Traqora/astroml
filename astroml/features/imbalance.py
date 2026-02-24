"""Compute imbalance / net flow ratio features.

This module provides a small, well-documented utility for computing the
net-flow imbalance between sent and received amounts. The primary output is
the "net flow ratio" defined as (sent - received) / (sent + received), which
is bounded in [-1, 1].

A `log_scale` option lets callers work in log-space (useful when amounts span
many orders of magnitude). Inputs may be scalars, NumPy arrays, or pandas
Series; outputs preserve the input shape and type where reasonable.
"""
from typing import Union, Optional

import numpy as np
import pandas as pd

Number = Union[float, int]
ArrayLike = Union[Number, np.ndarray, pd.Series, list, tuple]


def net_flow_ratio(
    sent: ArrayLike,
    received: ArrayLike,
    *,
    log_scale: bool = False,
    log_base: float = 10.0,
    eps: float = 1e-9,
) -> Union[float, np.ndarray, pd.Series]:
    """Compute net flow ratio = (sent - received) / (sent + received).

    Args:
        sent: Sent amounts (scalar or array-like).
        received: Received amounts (same shape as `sent`).
        log_scale: If True, apply log scaling to amounts before computing
            the ratio. This is `log_base(log(amount + eps))`.
        log_base: Base for logarithm when `log_scale` is True.
        eps: Small epsilon added to amounts before logging to avoid -inf.

    Returns:
        Ratio in [-1, 1]. Returns a scalar if inputs were scalars, a
        NumPy array for array-like inputs, or a pandas Series if a Series was
        provided.

    Notes:
        - If sent + received == 0, the ratio is defined to be 0.
        - Use `log_scale=True` to compress dynamic ranges; results are still
          normalized into [-1, 1].

    Examples:
        >>> net_flow_ratio(100, 40)
        0.42857142857142855
        >>> net_flow_ratio([1, 0], [0, 1])
        array([1., -1.])
    """
    # Helper to convert inputs while preserving type information for output
    sent_is_series = isinstance(sent, pd.Series)
    recv_is_series = isinstance(received, pd.Series)

    # Convert to numpy arrays
    sent_arr = np.asarray(sent, dtype=float)
    recv_arr = np.asarray(received, dtype=float)

    if sent_arr.shape != recv_arr.shape:
        raise ValueError("`sent` and `received` must have the same shape")

    if log_scale:
        if log_base <= 0:
            raise ValueError("log_base must be positive")
        sent_arr = np.log(sent_arr + eps) / np.log(log_base)
        recv_arr = np.log(recv_arr + eps) / np.log(log_base)

    num = sent_arr - recv_arr
    den = sent_arr + recv_arr

    # Safe division: when denominator is zero, define ratio to be 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den == 0, 0.0, num / den)

    # If inputs were scalar numbers, return a scalar
    if np.isscalar(sent) or np.isscalar(received):
        return float(np.asarray(ratio).item())

    # If original input was a pandas Series, preserve index
    if sent_is_series or recv_is_series:
        # Use index of sent if available, else received
        idx = sent.index if sent_is_series else received.index
        return pd.Series(ratio, index=idx)

    return np.asarray(ratio)


def net_flow_ratio_from_transactions(
    df: pd.DataFrame,
    sent_col: str = "sent_amount",
    received_col: str = "received_amount",
    out_col: str = "net_flow_ratio",
    **kwargs,
) -> pd.DataFrame:
    """Compute `net_flow_ratio` for each row in a transactions DataFrame.

    This convenience function adds a new column to the DataFrame (a copy is
    returned) containing the computed ratio.
    """
    if sent_col not in df or received_col not in df:
        raise KeyError(f"DataFrame must contain '{sent_col}' and '{received_col}' columns")

    out = df.copy()
    out[out_col] = net_flow_ratio(out[sent_col], out[received_col], **kwargs)
    return out
