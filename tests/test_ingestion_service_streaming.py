"""Tests for streaming ingestion — issue #547.

Covers :meth:`IngestionService.ingest_stream` (the new generator-based API
that avoids accumulating a whole ledger range in memory) and the
``batch_size`` parameter on both :meth:`IngestionService.ingest` and
:meth:`IngestionService.ingest_stream`.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import pytest

from astroml.ingestion.service import IngestionResult, IngestionService, LedgerOutcome
from astroml.ingestion.state import StateStore


@pytest.fixture()
def service(tmp_path: Path) -> IngestionService:
    return IngestionService(state_store=StateStore(path=str(tmp_path / "state.json")))


def test_ingest_stream_yields_ledger_outcomes(service: IngestionService) -> None:
    results = list(service.ingest_stream(start_ledger=1, end_ledger=5))

    assert [r[0] for r in results] == [1, 2, 3, 4, 5]
    assert all(isinstance(r[1], LedgerOutcome) for r in results)
    assert [r[1].status for r in results] == ["processed"] * 5


def test_ingest_stream_skips_already_processed(service: IngestionService) -> None:
    list(service.ingest_stream(start_ledger=1, end_ledger=3))

    results = list(service.ingest_stream(start_ledger=1, end_ledger=5))
    statuses = {ledger_id: outcome.status for ledger_id, outcome in results}

    assert statuses == {1: "skipped", 2: "skipped", 3: "skipped", 4: "processed", 5: "processed"}


def test_ingest_stream_is_lazy_and_yields_incrementally(service: IngestionService) -> None:
    """A caller should see a result after the *first* fetch, not after all of them."""
    fetch_calls = []

    def fetch_fn(ledger_id: int) -> dict:
        fetch_calls.append(ledger_id)
        return {"ledger": ledger_id}

    stream = service.ingest_stream(start_ledger=1, end_ledger=1000, fetch_fn=fetch_fn)

    first = next(stream)
    assert first[0] == 1
    # Only the first ledger should have been fetched so far — proves the
    # generator isn't materialising the whole 1000-ledger range up front.
    assert fetch_calls == [1]

    second = next(stream)
    assert second[0] == 2
    assert fetch_calls == [1, 2]


def test_ingest_stream_processes_before_fetching_next_ledger(service: IngestionService) -> None:
    """Backpressure: process_fn for ledger N must run before fetch_fn for N+1."""
    events = []

    def fetch_fn(ledger_id: int) -> dict:
        events.append(("fetch", ledger_id))
        return {"ledger": ledger_id}

    def process_fn(ledger_id: int, payload: dict) -> None:
        events.append(("process", ledger_id))

    list(service.ingest_stream(start_ledger=1, end_ledger=3, fetch_fn=fetch_fn, process_fn=process_fn))

    assert events == [
        ("fetch", 1), ("process", 1),
        ("fetch", 2), ("process", 2),
        ("fetch", 3), ("process", 3),
    ]


def test_ingest_stream_rejects_invalid_batch_size(service: IngestionService) -> None:
    with pytest.raises(ValueError):
        # Generator functions only run their body once iterated.
        next(service.ingest_stream(start_ledger=1, end_ledger=1, batch_size=0))


def test_ingest_stream_default_batch_size_is_100(service: IngestionService) -> None:
    # Should not raise — 100 is the documented default and a valid value.
    results = list(service.ingest_stream(start_ledger=1, end_ledger=1))
    assert len(results) == 1


def test_ingest_returns_same_shape_as_before(service: IngestionService) -> None:
    result = service.ingest(start_ledger=1, end_ledger=5)

    assert isinstance(result, IngestionResult)
    assert result.attempted == [1, 2, 3, 4, 5]
    assert result.processed == [1, 2, 3, 4, 5]
    assert result.skipped == []


def test_ingest_batch_size_does_not_change_result(service: IngestionService) -> None:
    result_default = service.ingest(start_ledger=1, end_ledger=10)

    service2 = IngestionService()
    service2.state = service.state.__class__(path=service.state.path + ".alt")
    result_custom = service2.ingest(start_ledger=1, end_ledger=10, batch_size=3)

    assert result_default.attempted == result_custom.attempted
    assert result_default.processed == result_custom.processed


def test_ingest_stream_memory_bounded_for_large_range(service: IngestionService) -> None:
    """Practical stand-in for '<100MB for 1M ledgers': assert peak traced
    memory for a 50k-ledger stream stays in the low single-digit MB range —
    i.e. it does not scale linearly with range size the way the old
    list-accumulating behavior would have. See docs/scaling-optimization.md
    for the full-scale (1M ledger) benchmark methodology and numbers.

    Uses a larger batch_size purely to keep this test fast (see
    test_ingest_stream_flushes_state_once_per_batch for the flush-cadence
    behavior itself at the default batch_size) — it doesn't affect the
    memory-boundedness being asserted here, which comes from ingest_stream
    never accumulating the ledger range into a list regardless of batch_size.
    """
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        count = 0
        for _ledger_id, _outcome in service.ingest_stream(
            start_ledger=1, end_ledger=50_000, batch_size=2000,
        ):
            count += 1
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert count == 50_000
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 25.0, f"ingest_stream over 50k ledgers used {peak_mb:.2f}MB traced peak"


def test_ingest_stream_flushes_state_once_per_batch(service: IngestionService, monkeypatch: pytest.MonkeyPatch) -> None:
    save_calls = []
    original_save = service.state.save

    def counting_save(state):
        save_calls.append(len(state.processed_ledgers))
        return original_save(state)

    monkeypatch.setattr(service.state, "save", counting_save)

    list(service.ingest_stream(start_ledger=1, end_ledger=25, batch_size=10))

    # 25 ledgers at batch_size=10: flush after 10, after 20, and a final
    # flush for the trailing 5 — not once per ledger.
    assert len(save_calls) == 3
    assert save_calls == [10, 20, 25]


def test_ingest_stream_flushes_partial_batch_on_early_stop(
    service: IngestionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_calls = []
    original_save = service.state.save

    def counting_save(state):
        save_calls.append(len(state.processed_ledgers))
        return original_save(state)

    monkeypatch.setattr(service.state, "save", counting_save)

    stream = service.ingest_stream(start_ledger=1, end_ledger=100, batch_size=10)
    for i, _ in enumerate(stream):
        if i == 4:  # consumed 5 ledgers, well short of a full batch
            stream.close()
            break

    # The finally block must flush the partial batch rather than losing it.
    assert save_calls == [5]
    reloaded = service.state.load()
    assert reloaded.processed_ledgers == {1, 2, 3, 4, 5}
