from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
import bisect
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from ...cache import cached_graph_snapshot



# Issue #199 — default chunk size for the streaming graph builder. SQLAlchemy
# fetches rows from the DB in batches of this many; the iterator yields each
# edge individually so callers never see a fully-materialised window list.
DEFAULT_STREAM_CHUNK_SIZE = 5_000


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

@cached_graph_snapshot(ttl_seconds=1800)
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

    # Issue #546 — skip the defensive copy when the caller already handed us
    # a list; `list(edges)` on an already-materialised list still allocates
    # a full second copy, which is pure overhead at graph sizes where memory
    # is the concern in the first place.
    if presorted:
        sorted_edges = edges if isinstance(edges, list) else list(edges)
    else:
        sorted_edges = _ensure_sorted_by_ts(edges)

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

@cached_graph_snapshot(ttl_seconds=1800)
def snapshot_last_n_days(
    edges: Sequence[Edge],
    now_ts: int,
    days: int = 30,
    presorted: bool = True,
) -> Tuple[Set[str], List[Edge]]:
    """Convenience wrapper to extract last N days window inclusive of now_ts.

    - days: configurable window size in days (>=1)
    - now_ts: anchor timestamp (epoch seconds)

    The window uses inclusive bounds on both sides: [start_ts, now_ts].
    The start bound is therefore computed as now_ts - days*86400 so events that
    land exactly on the cutoff are included.
    Example: days=1 -> [now_ts-86400, now_ts].
    """
    if days <= 0:
        raise ValueError("days must be >= 1")
    seconds = days * 86400
    start_ts = now_ts - seconds
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


@dataclass(frozen=True)
class SnapshotMeta:
    """Window metadata without the edge payload — issue #199.

    Yielded alongside a fresh edge iterator by
    :func:`iter_db_snapshot_edges` so callers can decide how (or whether)
    to buffer the edges, instead of being forced to hold a fully-built
    ``List[Edge]`` in RAM.
    """

    index: int
    start: datetime
    end: datetime


def iter_db_snapshot_edges(
    window: str = "7d",
    t0: Optional[datetime] = None,
    t_now: Optional[datetime] = None,
    step: Optional[str] = None,
    session=None,
    chunk_size: int = DEFAULT_STREAM_CHUNK_SIZE,
) -> Generator[Tuple["SnapshotMeta", Iterator["Edge"]], None, None]:
    """Streaming variant of :func:`iter_db_snapshots` — issue #199.

    Each yielded ``(meta, edges)`` pair gives the window bounds plus a
    fresh generator that pulls rows from the database in chunks of
    ``chunk_size`` via SQLAlchemy's ``yield_per`` and converts each row
    into an :class:`Edge` lazily. Peak memory per window is bounded by
    ``chunk_size`` (default 5 000 edges ≈ a few MB) regardless of how
    many edges the window actually contains.

    The edge iterator MUST be drained or discarded before advancing to
    the next ``(meta, edges)`` pair — the underlying SQLAlchemy result
    will be reused. The function does not yield a ``nodes`` set; build
    it incrementally if you need one.

    Use this in place of :func:`iter_db_snapshots` whenever a window may
    plausibly contain enough edges to risk OOM on the training machine.
    """
    from astroml.db.schema import NormalizedTransaction
    from sqlalchemy import func as sqlfunc, select

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

        result = session.execute(
            select(
                NormalizedTransaction.sender,
                NormalizedTransaction.receiver,
                NormalizedTransaction.timestamp,
            )
            .where(
                NormalizedTransaction.timestamp >= window_start,
                NormalizedTransaction.timestamp <= window_end,
                NormalizedTransaction.receiver.isnot(None),
                NormalizedTransaction.sender != NormalizedTransaction.receiver,
            )
            .order_by(NormalizedTransaction.timestamp)
            .execution_options(yield_per=chunk_size, stream_results=True)
        )

        def _edges_iter(_result=result) -> Iterator[Edge]:
            for row in _result:
                yield Edge(
                    src=row.sender,
                    dst=row.receiver,
                    timestamp=int(row.timestamp.timestamp()),
                )

        yield (
            SnapshotMeta(index=index, start=window_start, end=window_end),
            _edges_iter(),
        )

        window_start += step_delta
        index += 1


def _build_snapshot_window(
    index: int,
    window_start: datetime,
    window_end: datetime,
    chunk_size: int,
) -> SnapshotWindow:
    """Build a single snapshot window from the database."""
    from astroml.db.schema import NormalizedTransaction
    from astroml.db.session import get_session
    from sqlalchemy import select

    session = get_session()
    try:
        result = session.execute(
            select(
                NormalizedTransaction.sender,
                NormalizedTransaction.receiver,
                NormalizedTransaction.timestamp,
            )
            .where(
                NormalizedTransaction.timestamp >= window_start,
                NormalizedTransaction.timestamp <= window_end,
                NormalizedTransaction.receiver.isnot(None),
                NormalizedTransaction.sender != NormalizedTransaction.receiver,
            )
            .order_by(NormalizedTransaction.timestamp)
        )

        edges: List[Edge] = []
        nodes: Set[str] = set()

        for row in result.yield_per(chunk_size):
            edge = Edge(
                src=row.sender,
                dst=row.receiver,
                timestamp=int(row.timestamp.timestamp()),
            )
            edges.append(edge)
            nodes.add(edge.src)
            nodes.add(edge.dst)

        # Issue #546 — drop the SQLAlchemy result/cursor buffers before
        # allocating the returned SnapshotWindow so the two aren't briefly
        # alive together at peak.
        del result

        return SnapshotWindow(
            index=index,
            start=window_start,
            end=window_end,
            edges=edges,
            nodes=nodes,
        )
    finally:
        session.close()


def iter_db_snapshots(
    window: str = "7d",
    t0: Optional[datetime] = None,
    t_now: Optional[datetime] = None,
    step: Optional[str] = None,
    session=None,
    chunk_size: int = 100_000,
    workers: int = 1,
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
        chunk_size: Number of rows to stream per fetch from the DB. Larger values
            reduce round-trips but increase peak memory; smaller values keep the
            working set bounded for long-window snapshots.
        workers: Number of concurrent window fetch workers. Set to >1 to prefetch
            windows in parallel when using the default session factory.

    Yields:
        :class:`SnapshotWindow` instances in chronological order.
    """
    from astroml.db.schema import NormalizedTransaction
    from sqlalchemy import select, func as sqlfunc

    session_provided = session is not None
    if session is None:
        from astroml.db.session import get_session
        session = get_session()

    win_delta = _parse_window_size(window)
    step_delta = _parse_window_size(step) if step else win_delta

    if chunk_size is None or chunk_size <= 0:
        chunk_size = 100_000

    if t_now is None:
        t_now = datetime.now(timezone.utc)

    if t0 is None:
        result = session.execute(
            select(sqlfunc.min(NormalizedTransaction.timestamp))
        ).scalar()
        if result is None:
            session.close()
            return  # empty DB
        t0 = result if result.tzinfo else result.replace(tzinfo=timezone.utc)

    if t_now.tzinfo is None:
        t_now = t_now.replace(tzinfo=timezone.utc)
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)

    window_start = t0
    index = 0

    if workers > 1 and not session_provided:
        session.close()

        pending_windows: Dict[int, SnapshotWindow] = {}
        futures: Dict[int, "Future[SnapshotWindow]"] = {}
        next_index_to_yield = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            while window_start < t_now or futures:
                while window_start < t_now and len(futures) < workers:
                    window_end = min(window_start + win_delta, t_now)
                    future = executor.submit(
                        _build_snapshot_window,
                        index,
                        window_start,
                        window_end,
                        chunk_size,
                    )
                    futures[index] = future
                    window_start += step_delta
                    index += 1

                if not futures:
                    break

                done, _ = wait(set(futures.values()), return_when=FIRST_COMPLETED)
                for future in done:
                    result_window = future.result()
                    pending_windows[result_window.index] = result_window
                    future_index = next(
                        idx for idx, fut in futures.items() if fut is future
                    )
                    del futures[future_index]

                while next_index_to_yield in pending_windows:
                    yield pending_windows.pop(next_index_to_yield)
                    next_index_to_yield += 1

        return

    while window_start < t_now:
        window_end = min(window_start + win_delta, t_now)

        result = session.execute(
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
        )

        edges: List[Edge] = []
        nodes: Set[str] = set()

        # Stream rows in chunks to keep the working set bounded even for long
        # windows. This avoids pulling the full result set into memory at once.
        for row in result.yield_per(chunk_size):
            edge = Edge(
                src=row.sender,
                dst=row.receiver,
                timestamp=int(row.timestamp.timestamp()),
            )
            edges.append(edge)
            nodes.add(edge.src)
            nodes.add(edge.dst)

        del result  # Issue #546 — drop cursor buffers before the next window.

        yield SnapshotWindow(
            index=index,
            start=window_start,
            end=window_end,
            edges=edges,
            nodes=nodes,
        )

        window_start += step_delta
        index += 1
