"""Lightweight memory profiling for graph-building operations — issue #546.

Deliberately dependency-light: only ``tracemalloc`` (stdlib) and ``psutil``
(already an astroml dependency used elsewhere, e.g. :mod:`astroml.benchmarking.utils`).
``astroml.benchmarking`` is *not* reused here because it unconditionally imports
``torch`` at module load time, which would drag a heavy, unrelated dependency into
every caller of :mod:`astroml.features.graph`.
"""

from __future__ import annotations

import functools
import logging
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import psutil

logger = logging.getLogger("astroml.features.graph.memory_profile")

_F = TypeVar("_F", bound=Callable[..., Any])

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class GraphMemoryProfile:
    """Memory/time footprint of a single graph-building call.

    Attributes:
        n_nodes: Number of distinct nodes touched by the profiled call, if known.
        n_edges: Number of edges touched by the profiled call, if known.
        traced_peak_mb: Peak Python-allocated memory during the call, from
            ``tracemalloc`` (isolates the call's own allocations from the rest
            of the process).
        rss_delta_mb: Change in process resident set size (RSS) across the
            call, from ``psutil`` (captures C-extension / non-Python
            allocations that ``tracemalloc`` cannot see, e.g. NumPy buffers).
        duration_s: Wall-clock duration of the call in seconds.
    """

    n_nodes: int
    n_edges: int
    traced_peak_mb: float
    rss_delta_mb: float
    duration_s: float

    def summary(self) -> str:
        return (
            f"nodes={self.n_nodes} edges={self.n_edges} "
            f"traced_peak={self.traced_peak_mb:.2f}MB "
            f"rss_delta={self.rss_delta_mb:.2f}MB "
            f"duration={self.duration_s:.3f}s"
        )


def profile_graph_memory(
    fn: Callable[..., Any],
    *args: Any,
    n_nodes: int = 0,
    n_edges: int = 0,
    **kwargs: Any,
) -> tuple[Any, GraphMemoryProfile]:
    """Run ``fn(*args, **kwargs)`` and measure its memory footprint.

    ``n_nodes``/``n_edges`` are caller-supplied labels for the resulting
    :class:`GraphMemoryProfile` (this function has no way to know what ``fn``
    actually builds) — pass the sizes you already know from the call site.

    Safe to nest: if ``tracemalloc`` is already running (e.g. an outer
    profiling call), the existing trace is left running and only the local
    peak is measured relative to the snapshot taken on entry.
    """
    process = psutil.Process(os.getpid())
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()

    rss_start = process.memory_info().rss
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        duration = time.perf_counter() - start
        _current, traced_peak = tracemalloc.get_traced_memory()
        rss_end = process.memory_info().rss
        if not already_tracing:
            tracemalloc.stop()

    profile = GraphMemoryProfile(
        n_nodes=n_nodes,
        n_edges=n_edges,
        traced_peak_mb=traced_peak / _BYTES_PER_MB,
        rss_delta_mb=(rss_end - rss_start) / _BYTES_PER_MB,
        duration_s=duration,
    )
    return result, profile


def memory_profiled(fn: _F) -> _F:
    """Decorator that logs a :class:`GraphMemoryProfile` for each call.

    Opt-in: the wrapped function only pays the profiling overhead when
    called with ``profile_memory=True`` (consumed by the wrapper and not
    forwarded to ``fn``); otherwise it behaves exactly like the undecorated
    function.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, profile_memory: bool = False, **kwargs: Any) -> Any:
        if not profile_memory:
            return fn(*args, **kwargs)

        result, profile = profile_graph_memory(fn, *args, **kwargs)
        logger.info("%s memory profile: %s", fn.__qualname__, profile.summary())
        return result

    return wrapper  # type: ignore[return-value]
