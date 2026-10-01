"""Regression tests for snapshot resource analysis (issue #984)."""

from datetime import datetime, timezone

from astroml.features.graph.snapshot import (
    EDGE_BYTES_ESTIMATE,
    NODE_BYTES_ESTIMATE,
    Edge,
    SnapshotWindow,
    analyze_snapshot_resources,
)

T = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _window(index: int, n_edges: int) -> SnapshotWindow:
    edges = [Edge(src=f"s{i}", dst=f"d{i}", timestamp=i) for i in range(n_edges)]
    nodes = {e.src for e in edges} | {e.dst for e in edges}
    return SnapshotWindow(index=index, start=T, end=T, edges=edges, nodes=nodes)


def test_empty_input():
    report = analyze_snapshot_resources([])
    assert report.window_count == 0
    assert report.total_edges == report.max_edges == 0
    assert report.peak_window_index is None
    assert report.estimated_peak_bytes == 0


def test_aggregates_counts_and_peak():
    report = analyze_snapshot_resources([_window(0, 2), _window(1, 5), _window(2, 1)])
    assert report.window_count == 3
    assert report.total_edges == 8
    assert report.total_nodes == 16
    assert report.max_edges == 5
    assert report.max_nodes == 10
    assert report.peak_window_index == 1
    assert report.estimated_peak_bytes == 5 * EDGE_BYTES_ESTIMATE + 10 * NODE_BYTES_ESTIMATE


def test_empty_windows_still_counted():
    report = analyze_snapshot_resources([_window(3, 0)])
    assert report.window_count == 1
    assert report.peak_window_index == 3
    assert report.estimated_peak_bytes == 0


def test_accepts_generator_output_as_list():
    windows = list(_window(i, i) for i in range(4))
    assert analyze_snapshot_resources(windows).peak_window_index == 3
