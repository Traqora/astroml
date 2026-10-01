"""Regression tests for snapshot orchestration retry (issue #979)."""

from datetime import datetime, timezone

import pytest

from astroml.features.graph import snapshot
from astroml.features.graph.snapshot import SnapshotWindow, _build_snapshot_window_with_retry

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 1, 2, tzinfo=timezone.utc)


def _window(index: int) -> SnapshotWindow:
    return SnapshotWindow(index=index, start=START, end=END, edges=[], nodes=set())


def test_retries_until_success(monkeypatch):
    calls = []

    def flaky(index, start, end, chunk_size):
        calls.append(index)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return _window(index)

    monkeypatch.setattr(snapshot, "_build_snapshot_window", flaky)
    result = _build_snapshot_window_with_retry(4, START, END, 10, max_attempts=3)
    assert result.index == 4
    assert len(calls) == 3


def test_raises_after_max_attempts(monkeypatch, caplog):
    calls = []

    def always_fail(index, start, end, chunk_size):
        calls.append(index)
        raise RuntimeError("down")

    monkeypatch.setattr(snapshot, "_build_snapshot_window", always_fail)
    with pytest.raises(RuntimeError, match="down"):
        _build_snapshot_window_with_retry(0, START, END, 10, max_attempts=2)
    assert len(calls) == 2
    assert sum("snapshot window build failed" in r.message for r in caplog.records) == 2


def test_no_retry_on_success(monkeypatch):
    calls = []

    def ok(index, start, end, chunk_size):
        calls.append(index)
        return _window(index)

    monkeypatch.setattr(snapshot, "_build_snapshot_window", ok)
    _build_snapshot_window_with_retry(1, START, END, 10)
    assert calls == [1]


def test_rejects_invalid_max_attempts():
    with pytest.raises(ValueError):
        _build_snapshot_window_with_retry(0, START, END, 10, max_attempts=0)
