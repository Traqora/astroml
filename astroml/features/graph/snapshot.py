from __future__ import annotations

import bisect
import logging
import time
import uuid
from collections.abc import Generator, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ...cache import cached_graph_snapshot

logger = logging.getLogger(__name__)

# Issue #199 — default chunk size for the streaming graph builder. SQLAlchemy
# fetches rows from the DB in batches of this many; the iterator yields each
# edge individually so callers never see a fully-materialised window list.
DEFAULT_STREAM_CHUNK_SIZE = 5_000

# RFC 7807 (application/problem+json) "type" base for this module's problems
# — issue #949. These raise sites previously raised bare ``ValueError``s
# with only a free-text message, so a caller that surfaced the error at an
# API boundary had nothing structured to render as a problem-details
# response. See :class:`SnapshotWindowError`.
_PROBLEM_TYPE_BASE = "https://astroml.dev/problems/graph-snapshot"


@dataclass(frozen=True)
class ProblemDetail:
    """RFC 7807 ``application/problem+json`` payload (issue #949).

    Field names and semantics follow RFC 7807 §3.1:

    - ``type``: a URI identifying the problem type (not necessarily
      dereferenceable); stable per distinct failure kind so clients can
      branch on it without parsing ``detail``.
    - ``title``: short, human-readable summary, constant per ``type``.
    - ``status``: the HTTP status code an API layer should map this to.
    - ``detail``: human-readable explanation specific to this occurrence.
    - ``instance``: a URI identifying this specific occurrence; defaults to
      a freshly generated URN so repeated failures are distinguishable in
      logs even without a request URL to anchor to.
    """

    type: str
    title: str
    status: int
    detail: str
    instance: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")

    def to_dict(self) -> dict[str, Any]:
        """Return the RFC 7807 member set as a plain dict, ready to serialize
        as ``application/problem+json``."""
        return {
            "type": self.type,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "instance": self.instance,
        }


class SnapshotWindowError(ValueError):
    """A graph-snapshot windowing error with an attached RFC 7807 problem detail.

    Subclasses :class:`ValueError` so existing ``except ValueError`` call
    sites (and bare ``except Exception``) keep working unchanged; the
    difference is ``problem`` now carries structured, machine-readable
    detail an API boundary can render as ``application/problem+json``
    instead of only a free-text ``str(exc)``.
    """

    def __init__(self, problem: ProblemDetail) -> None:
        super().__init__(problem.detail)
        self.problem = problem

    def to_problem_detail(self) -> dict[str, Any]:
        """Convenience accessor for ``self.problem.to_dict()``."""
        return self.problem.to_dict()


def _invalid_window_bounds_error(start_ts: int, end_ts: int) -> SnapshotWindowError:
    return SnapshotWindowError(
        ProblemDetail(
            type=f"{_PROBLEM_TYPE_BASE}/invalid-window-bounds",
            title="Invalid snapshot window bounds",
            status=400,
            detail=f"start_ts must be <= end_ts (got start_ts={start_ts}, end_ts={end_ts})",
        )
    )


def _invalid_day_count_error(days: int) -> SnapshotWindowError:
    return SnapshotWindowError(
        ProblemDetail(
            type=f"{_PROBLEM_TYPE_BASE}/invalid-day-count",
            title="Invalid snapshot day count",
            status=400,
            detail=f"days must be >= 1 (got days={days})",
        )
    )


def _unknown_window_unit_error(window: str, unit: str) -> SnapshotWindowError:
    return SnapshotWindowError(
        ProblemDetail(
            type=f"{_PROBLEM_TYPE_BASE}/unknown-window-unit",
            title="Unknown window size unit",
            status=400,
            detail=(
                f"Unknown window unit '{unit}' in window spec '{window}'. "
                "Use 'd' (days), 'h' (hours), or 's' (seconds)."
            ),
        )
    )


def _malformed_window_spec_error(window: str) -> SnapshotWindowError:
    return SnapshotWindowError(
        ProblemDetail(
            type=f"{_PROBLEM_TYPE_BASE}/malformed-window-spec",
            title="Malformed window size specification",
            status=400,
            detail=(
                f"Could not parse window spec '{window}'. Expected a numeric "
                "value followed by 'd' (days), 'h' (hours), or 's' (seconds), "
                "e.g. '7d', '24h', '3600s'."
            ),
        )
    )


# Issue #972 — bounded retry for the DB-backed window builder used by the
# ThreadPoolExecutor/joblib orchestration paths below (see "Parallel
# snapshot construction"). A single transient DB failure (e.g. a dropped
# connection) previously aborted the whole parallel backfill; retrying a
# few times with backoff lets the orchestration recover from blips instead
# of failing the entire run.
DEFAULT_SNAPSHOT_BUILD_MAX_RETRIES = 3
DEFAULT_SNAPSHOT_BUILD_RETRY_BASE_DELAY = 0.5  # seconds


def _mask_account_id(value: str) -> str:
    """Mask a Stellar account/address identifier for safe logging.

    Account identifiers are graph node labels and, per the project's PII
    handling standard, must never appear in full in logs. Keeps a short,
    non-identifying prefix/suffix for debuggability while redacting the
    middle of the value.
    """
    if not value:
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-4:]}"


@dataclass(frozen=True, repr=False)
class Edge:
    src: str
    dst: str
    # Epoch seconds for efficient comparisons; can be any monotonic numeric timestamp
    timestamp: int

    def __repr__(self) -> str:
        """Mask src/dst so accidentally logging an Edge can't leak account ids."""
        return (
            f"Edge(src={_mask_account_id(self.src)!r}, "
            f"dst={_mask_account_id(self.dst)!r}, timestamp={self.timestamp!r})"
        )


def _ensure_sorted_by_ts(edges: Sequence[Edge]) -> list[Edge]:
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
) -> tuple[set[str], list[Edge]]:
    """Return induced subgraph (nodes, edges) within [start_ts, end_ts] inclusive.

    - edges: sequence of Edge
    - start_ts/end_ts: inclusive window bounds (epoch seconds)
    - presorted: if True, assume edges are sorted by timestamp ascending; otherwise we will sort once.

    Efficiency:
      Uses binary search to find left/right indices and then slices, O(log N + K).
    """
    if start_ts > end_ts:
        raise _invalid_window_bounds_error(start_ts, end_ts)

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

    nodes: set[str] = set()
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
) -> tuple[set[str], list[Edge]]:
    """Convenience wrapper to extract last N days window inclusive of now_ts.

    - days: configurable window size in days (>=1)
    - now_ts: anchor timestamp (epoch seconds)

    The window uses inclusive bounds on both sides: [start_ts, now_ts].
    The start bound is therefore computed as now_ts - days*86400 so events that
    land exactly on the cutoff are included.
    Example: days=1 -> [now_ts-86400, now_ts].
    """
    if days <= 0:
        raise _invalid_day_count_error(days)
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

    index: int  # 0-based window index (t_0, t_1, …, t_now)
    start: datetime
    end: datetime
    edges: list[Edge]
    nodes: set[str]


def _parse_window_size(window: str) -> timedelta:
    """Parse a window size string like '7d', '24h', '3600s' into a timedelta.

    Raises:
        ValueError: if the string is empty/malformed or the size is not
            positive. Issue #991 — a zero or negative size made the snapshot
            iterators loop forever (``window_start += step`` never advanced).
    """
    if not isinstance(window, str) or len(window.strip()) < 2:
        raise ValueError(f"Invalid window size {window!r}. Use e.g. '7d', '24h', '3600s'.")
    window = window.strip()
    unit = window[-1].lower()
    try:
        value = int(window[:-1])
    except ValueError:
        raise ValueError(
            f"Invalid window size {window!r}. Use e.g. '7d', '24h', '3600s'."
        ) from None
    if value <= 0:
        raise ValueError(f"Window size must be positive, got {window!r}.")
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "s":
        return timedelta(seconds=value)
    raise _unknown_window_unit_error(window, unit)


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
    t0: datetime | None = None,
    t_now: datetime | None = None,
    step: str | None = None,
    session=None,
    chunk_size: int = DEFAULT_STREAM_CHUNK_SIZE,
) -> Generator[tuple[SnapshotMeta, Iterator[Edge]], None, None]:
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
    from sqlalchemy import func as sqlfunc
    from sqlalchemy import select

    from astroml.db.schema import NormalizedTransaction

    if session is None:
        from astroml.db.session import get_session

        session = get_session()

    win_delta = _parse_window_size(window)
    step_delta = _parse_window_size(step) if step else win_delta

    if t_now is None:
        t_now = datetime.now(timezone.utc)

    if t0 is None:
        result = session.execute(select(sqlfunc.min(NormalizedTransaction.timestamp))).scalar()
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
    max_retries: int = DEFAULT_SNAPSHOT_BUILD_MAX_RETRIES,
) -> SnapshotWindow:
    """Build a single snapshot window from the database.

    Retries up to ``max_retries`` times (issue #972) with exponential
    backoff if the query/build fails — e.g. a transient DB connection drop
    during a long-running parallel backfill — so one flaky window doesn't
    abort the whole orchestration run. Each failed attempt is logged with
    its attempt number and error context; once retries are exhausted the
    final exception is re-raised so genuine (non-transient) failures still
    surface to the caller.
    """
    from sqlalchemy import select

    from astroml.db.schema import NormalizedTransaction
    from astroml.db.session import get_session

    attempt = 0
    while True:
        attempt += 1
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

            edges: list[Edge] = []
            nodes: set[str] = set()

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
        except Exception as exc:
            if attempt > max_retries:
                logger.error(
                    "Snapshot window %d build failed after %d attempt(s), giving up: %s",
                    index,
                    attempt,
                    exc,
                )
                raise
            delay = DEFAULT_SNAPSHOT_BUILD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Snapshot window %d build failed (attempt %d/%d): %s; retrying in %.2fs",
                index,
                attempt,
                max_retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)
        finally:
            session.close()


SNAPSHOT_MAX_ATTEMPTS = 3


def _build_snapshot_window_with_retry(
    index: int,
    window_start: datetime,
    window_end: datetime,
    chunk_size: int,
    max_attempts: int = SNAPSHOT_MAX_ATTEMPTS,
) -> SnapshotWindow:
    """Build a snapshot window, retrying transient failures (issue #979).

    Each failed attempt is logged; the last exception is re-raised once
    ``max_attempts`` is exhausted.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return _build_snapshot_window(index, window_start, window_end, chunk_size)
        except Exception:
            logger.warning(
                "snapshot window build failed",
                extra={"window_index": index, "attempt": attempt, "max_attempts": max_attempts},
                exc_info=True,
            )
            if attempt == max_attempts:
                raise
    raise AssertionError("unreachable")


def iter_db_snapshots(
    window: str = "7d",
    t0: datetime | None = None,
    t_now: datetime | None = None,
    step: str | None = None,
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
    from sqlalchemy import func as sqlfunc
    from sqlalchemy import select

    from astroml.db.schema import NormalizedTransaction

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
        result = session.execute(select(sqlfunc.min(NormalizedTransaction.timestamp))).scalar()
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

        pending_windows: dict[int, SnapshotWindow] = {}
        futures: dict[int, Future[SnapshotWindow]] = {}
        next_index_to_yield = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            while window_start < t_now or futures:
                while window_start < t_now and len(futures) < workers:
                    window_end = min(window_start + win_delta, t_now)
                    future = executor.submit(
                        _build_snapshot_window_with_retry,
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
                    future_index = next(idx for idx, fut in futures.items() if fut is future)
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
            )
            .where(
                NormalizedTransaction.timestamp >= window_start,
                NormalizedTransaction.timestamp <= window_end,
                NormalizedTransaction.receiver.isnot(None),
                NormalizedTransaction.sender != NormalizedTransaction.receiver,
            )
            .order_by(NormalizedTransaction.timestamp)
        )

        edges: list[Edge] = []
        nodes: set[str] = set()

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


# ---------------------------------------------------------------------------
# Parallel snapshot construction — issue #768
# ---------------------------------------------------------------------------


def _build_snapshot_window_single(args: tuple) -> SnapshotWindow:
    """Build a single snapshot window — top-level for pickling by joblib."""
    index, window_start, window_end, chunk_size = args
    return _build_snapshot_window(index, window_start, window_end, chunk_size)


def parallel_build_snapshots(
    window: str = "7d",
    t0: datetime | None = None,
    t_now: datetime | None = None,
    step: str | None = None,
    chunk_size: int = 100_000,
    n_jobs: int = -1,
    backend: str = "loky",
    batch_size: int | None = None,
) -> list[SnapshotWindow]:
    """Build multiple snapshot windows in parallel using joblib.

    This speeds up snapshot construction for large time ranges by distributing
    independent window builds across CPU cores.  Determinism is preserved:
    results are returned in chronological order regardless of completion time.

    Args:
        window: Window size string, e.g. ``'7d'``, ``'24h'``, ``'3600s'``.
        t0: Start of the first window. Defaults to earliest DB timestamp.
        t_now: End of the last window. Defaults to ``datetime.now(UTC)``.
        step: Slide step between windows (defaults to ``window`` for
            non-overlapping).
        chunk_size: Rows to stream per DB fetch per window.
        n_jobs: Number of parallel workers. ``-1`` uses all CPUs.
        backend: joblib backend (``'loky'`` for processes, ``'threading'``
            for threads).
        batch_size: If set, process windows in batches of this many to
            cap peak memory. ``None`` (default) processes all at once.

    Returns:
        List of :class:`SnapshotWindow` instances in chronological order.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning(
            "joblib not installed — falling back to sequential snapshot build"
        )
        return list(
            iter_db_snapshots(
                window=window,
                t0=t0,
                t_now=t_now,
                step=step,
                chunk_size=chunk_size,
                workers=1,
            )
        )

    win_delta = _parse_window_size(window)
    step_delta = _parse_window_size(step) if step else win_delta

    # Resolve time bounds
    from sqlalchemy import func as sqlfunc
    from sqlalchemy import select

    from astroml.db.schema import NormalizedTransaction
    from astroml.db.session import get_session

    session = get_session()
    try:
        if t_now is None:
            t_now = datetime.now(timezone.utc)

        if t0 is None:
            result = session.execute(
                select(sqlfunc.min(NormalizedTransaction.timestamp))
            ).scalar()
            if result is None:
                return []  # empty DB
            t0 = result if result.tzinfo else result.replace(tzinfo=timezone.utc)

        if t_now.tzinfo is None:
            t_now = t_now.replace(tzinfo=timezone.utc)
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
    finally:
        session.close()

    # Pre-compute all window arguments
    windows_args: list[tuple[int, datetime, datetime, int]] = []
    window_start = t0
    index = 0
    while window_start < t_now:
        window_end = min(window_start + win_delta, t_now)
        windows_args.append((index, window_start, window_end, chunk_size))
        window_start += step_delta
        index += 1

    if not windows_args:
        return []

    logger.info(
        "Parallel snapshot build: %d windows, n_jobs=%d, batch_size=%s",
        len(windows_args),
        n_jobs,
        batch_size or "all",
    )

    # Optionally batch to cap peak memory
    if batch_size and batch_size > 0:
        all_results: list[SnapshotWindow] = []
        for batch_start in range(0, len(windows_args), batch_size):
            batch = windows_args[batch_start : batch_start + batch_size]
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_build_snapshot_window_single)(args) for args in batch
            )
            # Sort by index to preserve chronological order
            results.sort(key=lambda sw: sw.index)
            all_results.extend(results)
        return all_results

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_build_snapshot_window_single)(args) for args in windows_args
    )
    # Sort by index to preserve chronological order
    results.sort(key=lambda sw: sw.index)
    return results


def compute_node_features_parallel(
    node_ids: Sequence[str],
    compute_fn: Any,
    n_jobs: int = -1,
    backend: str = "loky",
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Compute node features in parallel for independent nodes.

    Each node's feature computation is independent, making this embarrassingly
    parallel.  Determinism is preserved because results are keyed by node ID.

    Args:
        node_ids: Sequence of node identifiers to compute features for.
        compute_fn: Callable ``(node_id) -> features`` that computes features
            for a single node.
        n_jobs: Number of parallel workers. ``-1`` uses all CPUs.
        backend: joblib backend.
        batch_size: Nodes per batch to limit peak memory.

    Returns:
        Dict mapping ``node_id -> computed features``.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not installed — falling back to sequential")
        return {nid: compute_fn(nid) for nid in node_ids}

    all_results: dict[str, Any] = {}
    node_list = list(node_ids)

    for batch_start in range(0, len(node_list), batch_size):
        batch = node_list[batch_start : batch_start + batch_size]
        features = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(compute_fn)(nid) for nid in batch
        )
        for nid, feat in zip(batch, features):
            all_results[nid] = feat

    return all_results


# ---------------------------------------------------------------------------
# Resource analysis — issue #984
# ---------------------------------------------------------------------------

# Rough per-object footprint used for capacity planning, not exact accounting.
EDGE_BYTES_ESTIMATE = 200
NODE_BYTES_ESTIMATE = 100


@dataclass(frozen=True)
class SnapshotResourceReport:
    """Aggregate resource usage across a set of snapshot windows."""

    window_count: int
    total_edges: int
    total_nodes: int
    max_edges: int
    max_nodes: int
    peak_window_index: int | None
    estimated_peak_bytes: int


def analyze_snapshot_resources(windows: Sequence[SnapshotWindow]) -> SnapshotResourceReport:
    """Summarise edge/node counts and estimated peak memory for ``windows``.

    Args:
        windows: Snapshot windows to analyse (may be empty).

    Returns:
        A :class:`SnapshotResourceReport`. ``peak_window_index`` is the
        ``index`` of the window with the largest estimated footprint, or
        ``None`` when ``windows`` is empty.
    """
    total_edges = total_nodes = max_edges = max_nodes = peak_bytes = 0
    peak_index: int | None = None
    for w in windows:
        n_edges, n_nodes = len(w.edges), len(w.nodes)
        total_edges += n_edges
        total_nodes += n_nodes
        max_edges = max(max_edges, n_edges)
        max_nodes = max(max_nodes, n_nodes)
        size = n_edges * EDGE_BYTES_ESTIMATE + n_nodes * NODE_BYTES_ESTIMATE
        if peak_index is None or size > peak_bytes:
            peak_bytes, peak_index = size, w.index

    report = SnapshotResourceReport(
        window_count=len(windows),
        total_edges=total_edges,
        total_nodes=total_nodes,
        max_edges=max_edges,
        max_nodes=max_nodes,
        peak_window_index=peak_index,
        estimated_peak_bytes=peak_bytes,
    )
    logger.debug("snapshot resource report", extra={"report": report})
    return report
