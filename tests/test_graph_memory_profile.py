"""Memory profiling tests for the graph module — issue #546.

Budgets below are deliberately generous (several times the empirically
observed peak on a reference machine — see docs/scaling-optimization.md for
the raw numbers) so they catch real regressions without being flaky across
Python builds/architectures.
"""

from __future__ import annotations

import pytest

from astroml.features.graph.memory_profile import (
    GraphMemoryProfile,
    memory_profiled,
    profile_graph_memory,
)
from astroml.features.graph.snapshot import Edge, window_snapshot

# MB budget per edge count, intentionally loose (see module docstring).
_MEMORY_BUDGET_MB = {
    1_000: 5.0,
    10_000: 20.0,
}


def _make_edges(n: int, start_ts: int = 1_700_000_000, step: int = 1) -> list[Edge]:
    edges = []
    ts = start_ts
    for i in range(n):
        edges.append(Edge(src=f"acct{i % 500}", dst=f"acct{(i + 1) % 500}", timestamp=ts))
        ts += step
    return edges


@pytest.mark.parametrize("size", [1000, 10000])
def test_graph_building_memory(size: int) -> None:
    edges = _make_edges(size)

    (_nodes, windowed_edges), profile = profile_graph_memory(
        window_snapshot,
        edges,
        edges[0].timestamp,
        edges[-1].timestamp,
        n_edges=size,
    )

    assert len(windowed_edges) == size
    assert profile.traced_peak_mb < _MEMORY_BUDGET_MB[size], (
        f"window_snapshot over {size} edges used {profile.traced_peak_mb:.2f}MB "
        f"traced peak, expected < {_MEMORY_BUDGET_MB[size]}MB"
    )


def test_profile_graph_memory_reports_duration_and_shape() -> None:
    edges = _make_edges(200)

    result, profile = profile_graph_memory(
        window_snapshot, edges, edges[0].timestamp, edges[-1].timestamp,
        n_nodes=42, n_edges=200,
    )

    assert result is not None
    assert isinstance(profile, GraphMemoryProfile)
    assert profile.n_nodes == 42
    assert profile.n_edges == 200
    assert profile.duration_s >= 0.0
    assert profile.traced_peak_mb >= 0.0


def test_profile_graph_memory_nests_without_stopping_outer_trace() -> None:
    import tracemalloc

    tracemalloc.start()
    try:
        edges = _make_edges(50)
        _result, profile = profile_graph_memory(
            window_snapshot, edges, edges[0].timestamp, edges[-1].timestamp,
        )
        assert profile.traced_peak_mb >= 0.0
        # The outer trace must still be running — nested profiling shouldn't
        # tear it down.
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_memory_profiled_decorator_is_opt_in() -> None:
    calls = []

    @memory_profiled
    def build(n: int) -> int:
        calls.append(n)
        return n * 2

    # Default path: no profiling overhead, behaves like the bare function.
    assert build(3) == 6
    # Opt-in path: profile_memory is consumed by the wrapper, not forwarded.
    assert build(4, profile_memory=True) == 8
    assert calls == [3, 4]


def test_graph_memory_profile_summary_contains_key_fields() -> None:
    profile = GraphMemoryProfile(
        n_nodes=10, n_edges=20, traced_peak_mb=1.5, rss_delta_mb=2.5, duration_s=0.01,
    )
    summary = profile.summary()
    assert "nodes=10" in summary
    assert "edges=20" in summary
    assert "1.50MB" in summary
