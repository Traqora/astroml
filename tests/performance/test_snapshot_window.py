"""Performance regression coverage for in-memory graph snapshots."""

from __future__ import annotations

import pytest


def test_window_snapshot_bisects_original_edge_sequence(monkeypatch):
    """Boundary searches must not allocate a separate timestamp array."""
    from astroml.features.graph import snapshot

    edges = [
        snapshot.Edge(src=f"node_{index}", dst="hub", timestamp=index)
        for index in range(100)
    ]
    bisected_sequences = []
    original_bisect_left = snapshot.bisect.bisect_left
    original_bisect_right = snapshot.bisect.bisect_right

    def recording_bisect_left(*args, **kwargs):
        bisected_sequences.append(args[0])
        return original_bisect_left(*args, **kwargs)

    def recording_bisect_right(*args, **kwargs):
        bisected_sequences.append(args[0])
        return original_bisect_right(*args, **kwargs)

    monkeypatch.setattr(snapshot.bisect, "bisect_left", recording_bisect_left)
    monkeypatch.setattr(snapshot.bisect, "bisect_right", recording_bisect_right)

    nodes, window_edges = snapshot.window_snapshot.__wrapped__(edges, 25, 74)

    assert bisected_sequences == [edges, edges]
    assert len(window_edges) == 50
    assert nodes == {edge.src for edge in window_edges} | {"hub"}


@pytest.mark.benchmark(group="graph-snapshot-window")
def test_window_snapshot_large_presorted_input_baseline(benchmark):
    """Record the cost of extracting a narrow window from a large edge list."""
    from astroml.features.graph.snapshot import Edge, window_snapshot

    edges = [
        Edge(src=f"node_{index}", dst="hub", timestamp=index)
        for index in range(20_000)
    ]

    nodes, window_edges = benchmark(
        window_snapshot.__wrapped__,
        edges,
        9_000,
        10_000,
        presorted=True,
    )

    assert len(window_edges) == 1_001
    assert len(nodes) == 1_002