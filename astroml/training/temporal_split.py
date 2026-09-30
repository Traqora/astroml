"""Temporal train/test split utilities for AstroML.

Ensures strict "past-trains-on-future" ordering with no data leakage for
both flat tabular data (pandas DataFrames) and graph edge data.

Three public entry-points:

* :func:`temporal_train_test_split` — DataFrame splitter (re-exported from
  :mod:`astroml.validation.leakage` for convenience).
* :func:`temporal_graph_split` — splits a sequence of
  :class:`~astroml.features.graph.snapshot.Edge` objects into train/test
  edge sets with strict temporal ordering.
* :class:`TemporalSplitter` — thin config-driven wrapper that dispatches
  to the correct function and validates the result.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Re-export DataFrame splitter so callers can import from one place
# ---------------------------------------------------------------------------
from astroml.validation.leakage import (  # noqa: F401
    LeakageError,
    temporal_train_test_split,
    validate_temporal_split,
)

try:
    from astroml.features.graph.snapshot import Edge
except ImportError:  # allow import without torch-geometric
    Edge = Any  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Graph temporal split
# ---------------------------------------------------------------------------


@dataclass
class GraphSplitResult:
    """Holds the output of :func:`temporal_graph_split`.

    Attributes:
        train_edges: Edges whose timestamp is strictly before the cutoff.
        test_edges: Edges whose timestamp is >= the cutoff (or the last
            ``test_ratio`` fraction when no explicit cutoff is given).
        cutoff: The timestamp value used as the boundary.
    """

    train_edges: list[Any]
    test_edges: list[Any]
    cutoff: Any


def temporal_graph_split(
    edges: Sequence[Any],
    *,
    cutoff: Any | None = None,
    train_ratio: float = 0.8,
    time_attr: str = "timestamp",
) -> GraphSplitResult:
    """Split graph edges into temporal train/test partitions.

    Edges are split so that **all training edges precede all test edges**
    in time — no future information leaks into training.

    Two modes:

    * **Cutoff mode** (``cutoff`` is provided): edges with
      ``edge.{time_attr} < cutoff`` → train; the rest → test.
    * **Ratio mode** (default): edges are sorted by *time_attr* and split
      at ``int(len(edges) * train_ratio)``.

    Args:
        edges: Sequence of objects with a numeric or comparable
            *time_attr* attribute (e.g.
            :class:`~astroml.features.graph.snapshot.Edge` instances).
        cutoff: Explicit temporal boundary.  When provided, *train_ratio*
            is ignored.
        train_ratio: Fraction of (sorted) edges assigned to training when
            no *cutoff* is given.  Must be in ``(0, 1)``.
        time_attr: Attribute name on each edge object used as the
            timestamp.  Defaults to ``"timestamp"``.

    Returns:
        :class:`GraphSplitResult` with ``train_edges``, ``test_edges``,
        and the resolved ``cutoff``.

    Raises:
        ValueError: If *edges* is empty, *train_ratio* is out of range, or
            any edge is missing *time_attr*.
        LeakageError: If the resulting partitions overlap temporally (only
            possible in cutoff mode if the caller supplies a degenerate
            cutoff).
    """
    edges = list(edges)
    if not edges:
        return GraphSplitResult(train_edges=[], test_edges=[], cutoff=cutoff)

    # Validate that every edge has the expected attribute.
    for e in edges:
        if not hasattr(e, time_attr):
            raise ValueError(f"Edge object {e!r} has no attribute '{time_attr}'")

    if cutoff is not None:
        train_edges = [e for e in edges if getattr(e, time_attr) < cutoff]
        test_edges = [e for e in edges if getattr(e, time_attr) >= cutoff]
        resolved_cutoff = cutoff
    else:
        if not (0 < train_ratio < 1):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        sorted_edges = sorted(edges, key=lambda e: getattr(e, time_attr))
        split_idx = int(len(sorted_edges) * train_ratio)
        train_edges = sorted_edges[:split_idx]
        test_edges = sorted_edges[split_idx:]
        # Resolved cutoff = first timestamp in the test set (or None if empty).
        resolved_cutoff = getattr(test_edges[0], time_attr) if test_edges else None

    # Warn on empty partitions.
    if not train_edges:
        warnings.warn(
            "train_edges is empty — cutoff may be before all edge timestamps",
            UserWarning,
            stacklevel=2,
        )
    if not test_edges:
        warnings.warn(
            "test_edges is empty — cutoff may be after all edge timestamps",
            UserWarning,
            stacklevel=2,
        )

    # Hard leakage check.
    if train_edges and test_edges:
        train_max = max(getattr(e, time_attr) for e in train_edges)
        test_min = min(getattr(e, time_attr) for e in test_edges)
        if train_max >= test_min:
            raise LeakageError(
                f"Temporal overlap in graph split: train max ({train_max}) "
                f">= test min ({test_min})"
            )

    return GraphSplitResult(
        train_edges=train_edges,
        test_edges=test_edges,
        cutoff=resolved_cutoff,
    )


def validate_graph_split(result: GraphSplitResult, time_attr: str = "timestamp") -> bool:
    """Assert that a :class:`GraphSplitResult` has no temporal overlap.

    Args:
        result: Output of :func:`temporal_graph_split`.
        time_attr: Attribute name used as the timestamp.

    Returns:
        ``True`` if the split is clean.

    Raises:
        LeakageError: If overlap is detected.
    """
    if not result.train_edges or not result.test_edges:
        return True

    train_max = max(getattr(e, time_attr) for e in result.train_edges)
    test_min = min(getattr(e, time_attr) for e in result.test_edges)

    if train_max >= test_min:
        raise LeakageError(
            f"Temporal overlap in graph split: train max ({train_max}) " f">= test min ({test_min})"
        )
    return True


# ---------------------------------------------------------------------------
# High-level config-driven splitter
# ---------------------------------------------------------------------------


class TemporalSplitter:
    """Config-driven temporal train/test splitter.

    Supports both DataFrame and graph-edge inputs.  Validates the result
    automatically and raises :exc:`LeakageError` on any detected overlap.

    Args:
        train_ratio: Default fraction for train set when no cutoff is given.
        cutoff: Optional explicit temporal boundary.  Overrides *train_ratio*
            when set.
        time_col: Column/attribute name used as timestamp.

    Example — DataFrame usage::

        splitter = TemporalSplitter(train_ratio=0.8, time_col="closed_at")
        train_df, test_df = splitter.split_dataframe(transactions_df)

    Example — Graph edge usage::

        splitter = TemporalSplitter(train_ratio=0.8)
        result = splitter.split_edges(edges)
        # result.train_edges, result.test_edges
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        cutoff: Any | None = None,
        time_col: str = "timestamp",
    ):
        if not (0 < train_ratio < 1):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        self.train_ratio = train_ratio
        self.cutoff = cutoff
        self.time_col = time_col

    # ------------------------------------------------------------------
    # DataFrame split
    # ------------------------------------------------------------------

    def split_dataframe(
        self,
        df: Any,  # pd.DataFrame
        time_col: str | None = None,
    ) -> tuple[Any, Any]:
        """Split a DataFrame temporally and validate the result.

        Args:
            df: Input ``pd.DataFrame``.
            time_col: Override the instance-level *time_col*.

        Returns:
            ``(train_df, test_df)`` tuple.

        Raises:
            LeakageError: If the resulting split has temporal overlap.
        """
        col = time_col or self.time_col
        train_df, test_df = temporal_train_test_split(
            df,
            col,
            cutoff=self.cutoff,
            train_ratio=self.train_ratio,
        )
        validate_temporal_split(train_df, test_df, col)
        return train_df, test_df

    # ------------------------------------------------------------------
    # Graph edge split
    # ------------------------------------------------------------------

    def split_edges(
        self,
        edges: Sequence[Any],
        time_attr: str | None = None,
    ) -> GraphSplitResult:
        """Split graph edges temporally and validate the result.

        Args:
            edges: Sequence of edge objects.
            time_attr: Override the instance-level *time_col*.

        Returns:
            :class:`GraphSplitResult`.

        Raises:
            LeakageError: If the resulting split has temporal overlap.
        """
        attr = time_attr or self.time_col
        result = temporal_graph_split(
            edges,
            cutoff=self.cutoff,
            train_ratio=self.train_ratio,
            time_attr=attr,
        )
        validate_graph_split(result, time_attr=attr)
        return result
