from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple
import bisect


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    # Epoch seconds for efficient comparisons; can be any monotonic numeric timestamp
    timestamp: int


def _ensure_sorted_by_ts(edges: Sequence[Edge]) -> List[Edge]:
    if len(edges) <= 1:
        return list(edges)
    # Fast path: check if already non-decreasing by timestamp
    is_sorted = all(edges[i].timestamp <= edges[i + 1].timestamp for i in range(len(edges) - 1))
    if is_sorted:
        return list(edges)
    return sorted(edges, key=lambda e: e.timestamp)


def window_snapshot(
    edges: Sequence[Edge],
    start_ts: int,
    end_ts: int,
    presorted: bool = True,
) -> Tuple[Set[str], List[Edge]]:
    """Return induced subgraph (nodes, edges) within [start_ts, end_ts] inclusive.

    - edges: sequence of Edge
    - start_ts/end_ts: inclusive window bounds (epoch seconds)
    - presorted: if True, assume edges are sorted by timestamp ascending; otherwise we will sort once.

    Efficiency:
      Uses binary search to find left/right indices and then slices, O(log N + K).
    """
    if start_ts > end_ts:
        raise ValueError("start_ts must be <= end_ts")

    sorted_edges = list(edges) if presorted else _ensure_sorted_by_ts(edges)

    # Build an array of timestamps for bisect, referencing the same order.
    ts = [e.timestamp for e in sorted_edges]

    # Left bound: first index with timestamp >= start_ts
    left = bisect.bisect_left(ts, start_ts)
    # Right bound: last index with timestamp <= end_ts -> use bisect_right and subtract 1
    right_exclusive = bisect.bisect_right(ts, end_ts)

    if left >= right_exclusive:
        return set(), []

    window_edges = sorted_edges[left:right_exclusive]

    nodes: Set[str] = set()
    for e in window_edges:
        nodes.add(e.src)
        nodes.add(e.dst)

    return nodes, window_edges


def snapshot_last_n_days(
    edges: Sequence[Edge],
    now_ts: int,
    days: int = 30,
    presorted: bool = True,
) -> Tuple[Set[str], List[Edge]]:
    """Convenience wrapper to extract last N days window inclusive of now_ts.

    - days: configurable window size in days (>=1)
    - now_ts: anchor timestamp (epoch seconds)

    The start bound is computed as now_ts - days*86400 + 1 to ensure the window
    covers exactly N calendar days worth of seconds if treating bounds as inclusive.
    Example: days=1 -> [now_ts-86399, now_ts].
    """
    if days <= 0:
        raise ValueError("days must be >= 1")
    seconds = days * 86400
    start_ts = now_ts - seconds + 1
    if start_ts < 0:
        start_ts = 0
    return window_snapshot(edges, start_ts, now_ts, presorted=presorted)
