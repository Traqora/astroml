from __future__ import annotations

from datetime import datetime, timezone

from astroml.features.graph.snapshot import Edge, iter_db_snapshots


class FakeResult:
    def __init__(self, rows):
        self._rows = rows
        self.yield_per_calls = 0

    def yield_per(self, size):
        self.yield_per_calls += 1
        assert size == 2
        return iter(self._rows)

    def all(self):
        raise AssertionError("iter_db_snapshots must stream rows in chunks")


class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.execute_calls = 0

    def execute(self, _query):
        self.execute_calls += 1
        return FakeResult(self._rows)


def test_iter_db_snapshots_streams_in_chunks():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t_now = t0.replace(hour=1)
    rows = [
        type("Row", (), {"sender": "a", "receiver": "b", "timestamp": t0})(),
        type("Row", (), {"sender": "c", "receiver": "d", "timestamp": t0.replace(minute=1)})(),
    ]

    session = FakeSession(rows)

    windows = list(iter_db_snapshots("1h", t0=t0, t_now=t_now, session=session, chunk_size=2))

    assert len(windows) == 1
    assert windows[0].edges == [
        Edge(src="a", dst="b", timestamp=int(t0.timestamp())),
        Edge(src="c", dst="d", timestamp=int(t0.replace(minute=1).timestamp())),
    ]
    assert session.execute_calls == 1


def test_iter_db_snapshots_parallel_prefetches_windows(monkeypatch):
    from datetime import timedelta
    from astroml.features.graph.snapshot import iter_db_snapshots

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t_now = t0 + timedelta(hours=2)

    class FakeResult:
        def __init__(self, rows, scalar_value=None):
            self._rows = rows
            self._scalar = scalar_value

        def yield_per(self, size):
            assert size == 2
            return iter(self._rows)

        def scalar(self):
            return self._scalar

    class FakeSession:
        def __init__(self, result):
            self._result = result
            self.closed = False

        def execute(self, _query):
            return self._result

        def close(self):
            self.closed = True

    windows_rows = [
        [type("Row", (), {"sender": "a", "receiver": "b", "timestamp": t0})()],
        [type("Row", (), {"sender": "c", "receiver": "d", "timestamp": t0 + timedelta(hours=1)})()],
    ]
    call_count = {"calls": 0}

    def fake_get_session():
        if call_count["calls"] == 0:
            result = FakeResult([], scalar_value=t0)
        else:
            window_index = call_count["calls"] - 1
            result = FakeResult(windows_rows[window_index])
        call_count["calls"] += 1
        return FakeSession(result)

    monkeypatch.setattr("astroml.db.session.get_session", fake_get_session)

    windows = list(iter_db_snapshots("1h", t0=t0, t_now=t_now, chunk_size=2, workers=2))

    assert len(windows) == 2
    assert windows[0].edges[0].src == "a"
    assert windows[1].edges[0].src == "c"
    assert call_count["calls"] == 3
