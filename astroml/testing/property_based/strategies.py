"""Hypothesis strategies for ML data generation.

Provides reusable Hypothesis strategies that generate realistic ML inputs
such as feature matrices, label arrays, and raw model output vectors.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

# ── Feature matrix strategies ─────────────────────────────────────────────────


def feature_matrix_strategy(
    min_samples: int = 1,
    max_samples: int = 50,
    min_features: int = 1,
    max_features: int = 20,
) -> st.SearchStrategy[np.ndarray]:
    """Return a strategy that draws 2-D float32 feature matrices.

    Args:
        min_samples: Minimum number of rows (samples).
        max_samples: Maximum number of rows (samples).
        min_features: Minimum number of columns (features).
        max_features: Maximum number of columns (features).

    Returns:
        A Hypothesis strategy producing ``np.ndarray`` of shape
        ``(n_samples, n_features)`` with dtype ``float32``.
    """
    return np_st.arrays(
        dtype=np.float32,
        shape=st.tuples(
            st.integers(min_value=min_samples, max_value=max_samples),
            st.integers(min_value=min_features, max_value=max_features),
        ),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def binary_label_strategy(n_samples: int) -> st.SearchStrategy[np.ndarray]:
    """Return a strategy that draws a binary label array of fixed length.

    Args:
        n_samples: Number of labels to generate.

    Returns:
        A Hypothesis strategy producing ``np.ndarray`` of shape
        ``(n_samples,)`` with dtype ``int64`` containing only 0 or 1.
    """
    return np_st.arrays(
        dtype=np.int64,
        shape=(n_samples,),
        elements=st.integers(min_value=0, max_value=1),
    )


def multiclass_label_strategy(
    n_samples: int,
    n_classes: int = 3,
) -> st.SearchStrategy[np.ndarray]:
    """Return a strategy that draws a multi-class label array of fixed length.

    Args:
        n_samples: Number of labels to generate.
        n_classes: Number of distinct class values (labels in ``[0, n_classes)``).

    Returns:
        A Hypothesis strategy producing ``np.ndarray`` of shape
        ``(n_samples,)`` with dtype ``int64``.
    """
    return np_st.arrays(
        dtype=np.int64,
        shape=(n_samples,),
        elements=st.integers(min_value=0, max_value=max(0, n_classes - 1)),
    )


def probability_output_strategy(
    n_samples: int,
    n_classes: int = 2,
) -> st.SearchStrategy[np.ndarray]:
    """Return a strategy that draws a probability output matrix.

    Each row sums to 1.0 (simulating softmax output).

    Args:
        n_samples: Number of rows (one per sample).
        n_classes: Number of columns (one per class).

    Returns:
        A Hypothesis strategy producing ``np.ndarray`` of shape
        ``(n_samples, n_classes)`` with dtype ``float64`` where each
        row sums to 1.0.
    """

    @st.composite
    def _draw(draw: st.DrawFn) -> np.ndarray:  # type: ignore[type-arg]
        raw = draw(
            np_st.arrays(
                dtype=np.float64,
                shape=(n_samples, n_classes),
                elements=st.floats(
                    min_value=0.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        row_sums = raw.sum(axis=1, keepdims=True)
        # When an entire row is zero, substitute a uniform distribution
        # so that every row sums to exactly 1.0.
        zero_rows = (row_sums == 0.0).squeeze(axis=1)
        if zero_rows.any():
            raw[zero_rows] = 1.0 / n_classes
            row_sums = raw.sum(axis=1, keepdims=True)
        return (raw / row_sums).astype(np.float64)

    return _draw()


def scalar_score_strategy(
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> st.SearchStrategy[float]:
    """Return a strategy that draws a single finite scalar score.

    Args:
        min_value: Minimum allowed score value.
        max_value: Maximum allowed score value.

    Returns:
        A Hypothesis strategy producing a ``float`` in
        ``[min_value, max_value]``.
    """
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
