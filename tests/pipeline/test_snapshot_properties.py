"""Property-based tests for graph snapshot construction (issue #707).

Uses Hypothesis to generate small client/source graphs and asserts
invariants of the rolling-window snapshot builder in
``astroml.features.graph.snapshot``:

* determinism — same inputs always produce the same snapshot;
* node/edge counts — nodes are exactly the endpoints of the returned edges,
  and the edge count matches the in-window input count;
* temporal ordering — every returned edge lands inside the inclusive window
  and the output preserves timestamp order;
* window validity — inverted bounds raise, degenerate inputs behave;
* builder agreement — ``presorted=True/False`` agree on sorted inputs, and
  ``snapshot_last_n_days`` agrees with ``window_snapshot`` on the same bounds.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from astroml.features.graph.snapshot import Edge, snapshot_last_n_days, window_snapshot

# Small alphabets force node overlap (self-loops, multi-edges, shared hubs),
# which is where snapshot builders usually break.
_node_ids = st.sampled_from(["alice", "bob", "carol", "dave"])
_timestamps = st.integers(min_value=0, max_value=200)


def _edges(**kwargs):
    return st.builds(
        Edge,
        src=_node_ids,
        dst=_node_ids,
        timestamp=_timestamps,
        **kwargs,
    )


_edge_lists = st.lists(_edges(), min_size=0, max_size=30)


def _windows():
    return st.tuples(
        st.integers(min_value=0, max_value=200),
        st.integers(min_value=0, max_value=200),
    ).map(lambda pair: (min(pair), max(pair)))


_settings = settings(
    max_examples=75,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


class TestWindowSnapshotProperties:
    @_settings
    @given(edges=_edge_lists, bounds=_windows())
    def test_determinism(self, edges, bounds):
        """Same inputs produce identical snapshots on repeat calls."""
        start_ts, end_ts = bounds
        first = window_snapshot(edges, start_ts, end_ts, presorted=False)
        second = window_snapshot(edges, start_ts, end_ts, presorted=False)
        assert first[0] == second[0]
        assert first[1] == second[1]

    @_settings
    @given(edges=_edge_lists, bounds=_windows())
    def test_edge_count_matches_in_window_inputs(self, edges, bounds):
        """Returned edges are exactly the inputs whose ts is in [start, end]."""
        start_ts, end_ts = bounds
        _, out_edges = window_snapshot(edges, start_ts, end_ts, presorted=False)
        expected = [e for e in edges if start_ts <= e.timestamp <= end_ts]
        assert len(out_edges) == len(expected)
        assert sorted(out_edges, key=lambda e: (e.timestamp, e.src, e.dst)) == sorted(
            expected, key=lambda e: (e.timestamp, e.src, e.dst)
        )

    @_settings
    @given(edges=_edge_lists, bounds=_windows())
    def test_nodes_are_exactly_edge_endpoints(self, edges, bounds):
        """Node set is exactly the union of src/dst over returned edges."""
        start_ts, end_ts = bounds
        nodes, out_edges = window_snapshot(edges, start_ts, end_ts, presorted=False)
        expected = {e.src for e in out_edges} | {e.dst for e in out_edges}
        assert nodes == expected

    @_settings
    @given(edges=_edge_lists, bounds=_windows())
    def test_temporal_ordering(self, edges, bounds):
        """Every returned edge is in-window and output is ts-ordered."""
        start_ts, end_ts = bounds
        _, out_edges = window_snapshot(edges, start_ts, end_ts, presorted=False)
        assert all(start_ts <= e.timestamp <= end_ts for e in out_edges)
        stamps = [e.timestamp for e in out_edges]
        assert stamps == sorted(stamps)

    @_settings
    @given(edges=_edge_lists, bounds=_windows())
    def test_presorted_flag_agrees_on_sorted_inputs(self, edges, bounds):
        """presorted=True/False agree whenever the input is already sorted."""
        start_ts, end_ts = bounds
        ordered = sorted(edges, key=lambda e: e.timestamp)
        via_flag = window_snapshot(ordered, start_ts, end_ts, presorted=True)
        via_sort = window_snapshot(ordered, start_ts, end_ts, presorted=False)
        assert via_flag[0] == via_sort[0]
        assert via_flag[1] == via_sort[1]

    @_settings
    @given(edges=_edge_lists)
    def test_empty_window_returns_empty_snapshot(self, edges):
        """A window beyond every timestamp yields no nodes and no edges."""
        if not edges:
            start_ts = end_ts = 0
        else:
            latest = max(e.timestamp for e in edges)
            start_ts, end_ts = latest + 1, latest + 100
        nodes, out_edges = window_snapshot(edges, start_ts, end_ts, presorted=False)
        assert nodes == set()
        assert out_edges == []

    @_settings
    @given(edges=_edge_lists)
    def test_single_point_window(self, edges):
        """A zero-width window keeps exactly the edges stamped at that point."""
        if not edges:
            return
        point = edges[0].timestamp
        _, out_edges = window_snapshot(edges, point, point, presorted=False)
        assert all(e.timestamp == point for e in out_edges)
        assert len(out_edges) == sum(1 for e in edges if e.timestamp == point)

    def test_inverted_bounds_raise(self):
        with pytest.raises(ValueError, match="start_ts must be <= end_ts"):
            window_snapshot([Edge(src="a", dst="b", timestamp=5)], 10, 5)

    def test_empty_input(self):
        nodes, out_edges = window_snapshot([], 0, 100)
        assert nodes == set()
        assert out_edges == []


class TestSnapshotLastNDaysProperties:
    @_settings
    @given(edges=_edge_lists, now_ts=_timestamps, days=st.integers(min_value=1, max_value=10))
    def test_agrees_with_window_snapshot_on_same_bounds(self, edges, now_ts, days):
        """The N-day helper matches window_snapshot over [now-86400*days, now]."""
        nodes, out_edges = snapshot_last_n_days(edges, now_ts, days=days, presorted=False)
        start_ts = max(0, now_ts - days * 86400)
        exp_nodes, exp_edges = window_snapshot(edges, start_ts, now_ts, presorted=False)
        assert nodes == exp_nodes
        assert out_edges == exp_edges

    def test_non_positive_days_raise(self):
        with pytest.raises(ValueError, match="days must be >= 1"):
            snapshot_last_n_days([], now_ts=100, days=0)
