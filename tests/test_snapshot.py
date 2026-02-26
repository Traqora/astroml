from __future__ import annotations

import random
from astroml.features.graph.snapshot import Edge, window_snapshot, snapshot_last_n_days


def make_edges(n: int, start_ts: int = 1, step: int = 60):
    # Create monotonically increasing timestamps separated by 'step' seconds
    edges = []
    ts = start_ts
    for i in range(n):
        edges.append(Edge(src=f"u{i%5}", dst=f"v{i%7}", timestamp=ts))
        ts += step
    return edges


def test_window_snapshot_inclusive_bounds():
    edges = make_edges(10, start_ts=1000, step=10)
    # timestamps: 1000, 1010, ..., 1090
    nodes, win = window_snapshot(edges, start_ts=1010, end_ts=1050, presorted=True)
    assert [e.timestamp for e in win] == [1010, 1020, 1030, 1040, 1050]
    all_nodes = set()
    for e in win:
        all_nodes.add(e.src)
        all_nodes.add(e.dst)
    assert nodes == all_nodes


def test_window_snapshot_empty_when_outside():
    edges = make_edges(5, start_ts=200, step=5)
    nodes, win = window_snapshot(edges, start_ts=10, end_ts=15, presorted=True)
    assert nodes == set()
    assert win == []


def test_window_snapshot_unsorted_input():
    edges = make_edges(8, start_ts=100, step=3)
    shuffled = list(edges)
    random.shuffle(shuffled)
    nodes_s, win_s = window_snapshot(shuffled, start_ts=106, end_ts=115, presorted=False)
    # Corresponding sorted timestamps in this range are 106, 109, 112, 115
    assert [e.timestamp for e in win_s] == [106, 109, 112, 115]
    # Ensure node set aligns with edges returned
    nodes_calc = set()
    for e in win_s:
        nodes_calc.add(e.src)
        nodes_calc.add(e.dst)
    assert nodes_s == nodes_calc


def test_snapshot_last_n_days_window():
    # Construct edges hourly over 4 days
    hours = 24 * 4
    step = 3600
    start_ts = 1_000_000
    edges = make_edges(hours, start_ts=start_ts, step=step)
    now_ts = start_ts + (hours - 1) * step  # last edge timestamp

    # last 2 days should include last 48 edges
    nodes, win = snapshot_last_n_days(edges, now_ts=now_ts, days=2, presorted=True)
    assert len(win) == 48
    # Validate boundaries inclusive
    assert win[0].timestamp == now_ts - (48 - 1) * step
    assert win[-1].timestamp == now_ts


def test_invalid_params():
    edges = make_edges(2)
    try:
        window_snapshot(edges, start_ts=10, end_ts=5, presorted=True)
        assert False, "expected ValueError for inverted bounds"
    except ValueError:
        pass

    try:
        snapshot_last_n_days(edges, now_ts=100, days=0, presorted=True)
        assert False, "expected ValueError for non-positive days"
    except ValueError:
        pass
