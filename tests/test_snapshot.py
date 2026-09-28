from __future__ import annotations

import random

import pytest

from astroml.features.graph.snapshot import (
    Edge,
    _build_snapshot_window,
    snapshot_last_n_days,
    window_snapshot,
)


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

    # last 2 days with inclusive bounds should include 49 hourly edges
    nodes, win = snapshot_last_n_days(edges, now_ts=now_ts, days=2, presorted=True)
    assert len(win) == 49
    # Validate boundaries inclusive
    assert win[0].timestamp == now_ts - 2 * 86400
    assert win[-1].timestamp == now_ts


def test_snapshot_last_n_days_includes_exact_cutoff_boundary():
    now_ts = 30 * 86400
    edges = [
        Edge(src="excluded", dst="x", timestamp=now_ts - 30 * 86400 - 1),
        Edge(src="cutoff", dst="y", timestamp=now_ts - 30 * 86400),
        Edge(src="inside", dst="z", timestamp=now_ts),
    ]

    nodes, win = snapshot_last_n_days(edges, now_ts=now_ts, days=30, presorted=True)

    assert [e.timestamp for e in win] == [now_ts - 30 * 86400, now_ts]
    assert {e.src for e in win} == {"cutoff", "inside"}
    assert nodes == {"cutoff", "inside", "y", "z"}


def test_snapshot_last_n_days_clamps_negative_start_to_zero():
    now_ts = 10
    edges = [
        Edge(src="zero", dst="a", timestamp=0),
        Edge(src="inside", dst="b", timestamp=10),
    ]

    _, win = snapshot_last_n_days(edges, now_ts=now_ts, days=30, presorted=True)

    assert [e.timestamp for e in win] == [0, 10]


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


# ---------------------------------------------------------------------------
# Issue #967 — Edge repr/str must never leak full account identifiers to logs
# ---------------------------------------------------------------------------


def test_edge_repr_masks_account_ids():
    src = "GABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890AB"
    dst = "GZYXWVUTSRQPONMLKJIHGFEDCBA0987654321ZY"
    edge = Edge(src=src, dst=dst, timestamp=100)

    rendered = repr(edge)

    assert src not in rendered
    assert dst not in rendered
    assert src[:4] in rendered and src[-4:] in rendered
    assert dst[:4] in rendered and dst[-4:] in rendered


def test_edge_str_matches_masked_repr():
    edge = Edge(src="GABCDEFGHIJKLMNOP", dst="GZYXWVUTSRQPONML", timestamp=1)

    # dataclasses fall back to __repr__ for __str__ when __str__ isn't
    # defined — accidentally logging an Edge via f"{edge}" must be just as
    # safe as logging repr(edge).
    assert str(edge) == repr(edge)
    assert "GABCDEFGHIJKLMNOP" not in str(edge)


def test_edge_repr_fully_masks_short_ids():
    edge = Edge(src="abc", dst="xy", timestamp=1)

    rendered = repr(edge)

    assert "abc" not in rendered
    assert "xy" not in rendered


def test_edge_equality_unaffected_by_masked_repr():
    # Regression guard: overriding __repr__ must not disturb the
    # dataclass-generated __eq__/__hash__ that callers rely on.
    a = Edge(src="alice", dst="bob", timestamp=1)
    b = Edge(src="alice", dst="bob", timestamp=1)
    assert a == b


# ---------------------------------------------------------------------------
# Issue #972 — bounded retry for the DB-backed window builder used by the
# ThreadPoolExecutor/joblib orchestration paths.
# ---------------------------------------------------------------------------


class _FakeRetryResult:
    def __init__(self, rows):
        self._rows = rows

    def yield_per(self, size):
        return iter(self._rows)


class _FlakySession:
    """Raises for the first ``fail_times`` execute() calls, then succeeds."""

    def __init__(self, fail_times, rows):
        self.fail_times = fail_times
        self.rows = rows
        self.execute_calls = 0
        self.close_calls = 0

    def execute(self, _query):
        self.execute_calls += 1
        if self.execute_calls <= self.fail_times:
            raise RuntimeError(f"transient db error #{self.execute_calls}")
        return _FakeRetryResult(self.rows)

    def close(self):
        self.close_calls += 1


def _make_row(sender, receiver, timestamp):
    return type("Row", (), {"sender": sender, "receiver": receiver, "timestamp": timestamp})()


def test_build_snapshot_window_retries_transient_failures(monkeypatch):
    from datetime import datetime, timezone

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0.replace(hour=1)
    rows = [_make_row("alice", "bob", t0)]

    session = _FlakySession(fail_times=2, rows=rows)
    monkeypatch.setattr("astroml.db.session.get_session", lambda: session)
    monkeypatch.setattr("astroml.features.graph.snapshot.time.sleep", lambda _seconds: None)

    window = _build_snapshot_window(0, t0, t1, chunk_size=10, max_retries=3)

    assert window.edges == [Edge(src="alice", dst="bob", timestamp=int(t0.timestamp()))]
    # 2 failed attempts + 1 successful attempt
    assert session.execute_calls == 3
    # Session is closed after every attempt, including failed ones.
    assert session.close_calls == 3


def test_build_snapshot_window_raises_after_exhausting_retries(monkeypatch):
    from datetime import datetime, timezone

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0.replace(hour=1)

    session = _FlakySession(fail_times=10, rows=[])
    monkeypatch.setattr("astroml.db.session.get_session", lambda: session)
    monkeypatch.setattr("astroml.features.graph.snapshot.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="transient db error #4"):
        _build_snapshot_window(0, t0, t1, chunk_size=10, max_retries=3)

    # 1 initial attempt + 3 retries = 4 total attempts, then give up.
    assert session.execute_calls == 4
    assert session.close_calls == 4


def test_build_snapshot_window_logs_each_retry_attempt(monkeypatch, caplog):
    from datetime import datetime, timezone

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = t0.replace(hour=1)
    rows = [_make_row("alice", "bob", t0)]

    session = _FlakySession(fail_times=1, rows=rows)
    monkeypatch.setattr("astroml.db.session.get_session", lambda: session)
    monkeypatch.setattr("astroml.features.graph.snapshot.time.sleep", lambda _seconds: None)

    with caplog.at_level("WARNING", logger="astroml.features.graph.snapshot"):
        _build_snapshot_window(0, t0, t1, chunk_size=10, max_retries=3)

    assert any("attempt 1/4" in record.message for record in caplog.records)
