from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Generator, Iterable, List, Optional, Sequence, Set, Tuple
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


# ---------------------------------------------------------------------------
# DB-backed time-windowed snapshot slicer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SnapshotWindow:
    """A discrete time window slice ready for training."""
    index: int          # 0-based window index (t_0, t_1, …, t_now)
    start: datetime
    end: datetime
    edges: List[Edge]
    nodes: Set[str]


def _parse_window_size(window: str) -> timedelta:
    """Parse a window size string like '7d', '24h', '3600s' into a timedelta."""
    unit = window[-1].lower()
    value = int(window[:-1])
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "s":
        return timedelta(seconds=value)
    raise ValueError(f"Unknown window unit '{unit}'. Use 'd', 'h', or 's'.")


def iter_db_snapshots(
    window: str = "7d",
    t0: Optional[datetime] = None,
    t_now: Optional[datetime] = None,
    step: Optional[str] = None,
    session=None,
) -> Generator[SnapshotWindow, None, None]:
    """Yield discrete time-windowed graph snapshots from the database.

    Slices ``normalized_transactions`` into non-overlapping (or rolling)
    windows from ``t0`` to ``t_now``, each of size ``window``.

    Args:
        window: Window size string, e.g. ``'7d'``, ``'24h'``, ``'3600s'``.
        t0: Start of the first window. Defaults to the earliest timestamp in DB.
        t_now: End of the last window. Defaults to ``datetime.now(UTC)``.
        step: Slide step between windows (defaults to ``window`` for non-overlapping).
              Set smaller than ``window`` for rolling windows.
        session: SQLAlchemy session. If None, one is created via ``get_session()``.

    Yields:
        :class:`SnapshotWindow` instances in chronological order.
    """
    from astroml.db.schema import NormalizedTransaction
    from sqlalchemy import select, func as sqlfunc

    if session is None:
        from astroml.db.session import get_session
        session = get_session()

    win_delta = _parse_window_size(window)
    step_delta = _parse_window_size(step) if step else win_delta

    if t_now is None:
        t_now = datetime.now(timezone.utc)

    if t0 is None:
        result = session.execute(
            select(sqlfunc.min(NormalizedTransaction.timestamp))
        ).scalar()
        if result is None:
            return  # empty DB
        t0 = result if result.tzinfo else result.replace(tzinfo=timezone.utc)

    if t_now.tzinfo is None:
        t_now = t_now.replace(tzinfo=timezone.utc)
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)

    window_start = t0
    index = 0

    while window_start < t_now:
        window_end = min(window_start + win_delta, t_now)

        rows = session.execute(
            select(
                NormalizedTransaction.sender,
                NormalizedTransaction.receiver,
                NormalizedTransaction.timestamp,
            ).where(
                NormalizedTransaction.timestamp >= window_start,
                NormalizedTransaction.timestamp <= window_end,
                NormalizedTransaction.receiver.isnot(None),
                NormalizedTransaction.sender != NormalizedTransaction.receiver,
            ).order_by(NormalizedTransaction.timestamp)
        ).all()

        edges = [
            Edge(src=r.sender, dst=r.receiver, timestamp=int(r.timestamp.timestamp()))
            for r in rows
        ]
        nodes: Set[str] = set()
        for e in edges:
            nodes.add(e.src)
            nodes.add(e.dst)

        yield SnapshotWindow(
            index=index,
            start=window_start,
            end=window_end,
            edges=edges,
            nodes=nodes,
        )

        window_start += step_delta
        index += 1
