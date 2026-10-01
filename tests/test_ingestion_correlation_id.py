"""Tests for correlation-id / tracing propagation through ingestion — issues #944, #950.

:class:`~astroml.ingestion.service.IngestionService` previously never touched
the request/trace-correlation contextvar exposed by
:mod:`astroml.utils.logging`, so log lines emitted while an ingestion run was
in flight (including from user-supplied ``fetch_fn``/``process_fn`` callbacks)
carried no ``request_id`` and couldn't be correlated back to a single run.

These tests cover:
- #944 (tracing propagation): a correlation id is present and stable for the
  full duration of one ``ingest_stream``/``ingest`` call, including inside
  caller-supplied callbacks, and is restored afterwards.
- #950 (request-id correlation): a caller-supplied correlation id (e.g. one
  already set by an upstream HTTP handler) is inherited rather than
  overwritten, and independent runs get distinct ids when none is inherited.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astroml.ingestion.service import IngestionService
from astroml.ingestion.state import StateStore
from astroml.utils.logging import CorrelationId, get_correlation_id


@pytest.fixture()
def service(tmp_path: Path) -> IngestionService:
    return IngestionService(state_store=StateStore(path=str(tmp_path / "state.json")))


def test_ingest_stream_sets_a_correlation_id_for_the_run(service: IngestionService) -> None:
    assert get_correlation_id() is None

    seen_ids: list[str | None] = []
    for _ledger_id, _outcome in service.ingest_stream(start_ledger=1, end_ledger=3):
        seen_ids.append(get_correlation_id())

    assert all(seen_ids)
    assert len(set(seen_ids)) == 1, "correlation id must stay stable across the whole run"


def test_ingest_stream_restores_prior_correlation_id_after_completion(
    service: IngestionService,
) -> None:
    with CorrelationId("outer-request"):
        list(service.ingest_stream(start_ledger=1, end_ledger=3))
        assert get_correlation_id() == "outer-request"

    assert get_correlation_id() is None


def test_ingest_stream_propagates_correlation_id_into_fetch_and_process_callbacks(
    service: IngestionService,
) -> None:
    """#944 — tracing propagation: user callbacks running inside the stream
    must observe the same correlation id as the driving generator, so any
    logging they do is attributable to this ingestion run."""
    observed: list[str | None] = []

    def fetch_fn(ledger_id: int) -> dict:
        observed.append(get_correlation_id())
        return {"ledger": ledger_id}

    def process_fn(ledger_id: int, payload: dict) -> None:
        observed.append(get_correlation_id())

    with CorrelationId("trace-abc-123"):
        list(
            service.ingest_stream(
                start_ledger=1, end_ledger=2, fetch_fn=fetch_fn, process_fn=process_fn
            )
        )

    assert observed == ["trace-abc-123"] * 4


def test_ingest_stream_inherits_caller_correlation_id_instead_of_overwriting(
    service: IngestionService,
) -> None:
    """#950 — request-id correlation: when the caller already established a
    correlation id (e.g. an inbound HTTP request), ingestion must reuse it
    rather than minting a new, unrelated one."""
    with CorrelationId("caller-request-id"):
        seen = []
        for _ledger_id, _outcome in service.ingest_stream(start_ledger=1, end_ledger=1):
            seen.append(get_correlation_id())

    assert seen == ["caller-request-id"]


def test_ingest_stream_generates_distinct_ids_for_independent_runs_without_inheritance(
    service: IngestionService,
) -> None:
    list(service.ingest_stream(start_ledger=1, end_ledger=1))
    first_id = None
    for _ledger_id, _outcome in service.ingest_stream(start_ledger=2, end_ledger=2):
        first_id = get_correlation_id()

    service2 = IngestionService(state_store=StateStore(path=service.state.path + ".alt"))
    second_id = None
    for _ledger_id, _outcome in service2.ingest_stream(start_ledger=1, end_ledger=1):
        second_id = get_correlation_id()

    assert first_id is not None
    assert second_id is not None
    assert first_id != second_id


def test_ingest_propagates_correlation_id_via_ingest_stream(service: IngestionService) -> None:
    """`ingest()` delegates to `ingest_stream()`, so it must inherit the same
    propagation behavior without any extra wiring."""
    observed: list[str | None] = []

    def process_fn(ledger_id: int, payload: dict) -> None:
        observed.append(get_correlation_id())

    with CorrelationId("outer-ingest-call"):
        result = service.ingest(start_ledger=1, end_ledger=3, process_fn=process_fn)

    assert result.processed == [1, 2, 3]
    assert observed == ["outer-ingest-call"] * 3


def test_ingest_stream_correlation_id_restored_even_on_early_close(
    service: IngestionService,
) -> None:
    """Closing the generator early (GeneratorExit) must still unwind the
    correlation-id scope via the context manager, leaving no leaked state."""
    stream = service.ingest_stream(start_ledger=1, end_ledger=100)
    next(stream)
    stream.close()

    assert get_correlation_id() is None


def test_ingest_stream_correlation_id_restored_on_error(service: IngestionService) -> None:
    def fetch_fn(ledger_id: int) -> dict:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        list(service.ingest_stream(start_ledger=1, end_ledger=1, fetch_fn=fetch_fn))

    assert get_correlation_id() is None


def test_ingest_backfill_chunked_uses_one_correlation_id_across_all_chunks(
    service: IngestionService,
) -> None:
    """#957 — each chunk previously delegated to ``ingest_stream`` with no
    shared correlation-id scope around the loop, so ``ingest_stream`` minted a
    *fresh* id per chunk whenever the caller hadn't already set one. A single
    backfill spanning several chunks therefore split across multiple
    unrelated ``request_id``s in the logs, defeating #944/#950's tracing for
    the run that most needs it."""
    observed: list[str | None] = []

    def process_fn(ledger_id: int, payload: object) -> None:
        observed.append(get_correlation_id())

    list(
        service.ingest_backfill_chunked(
            start_ledger=1, end_ledger=6, chunk_size=2, process_fn=process_fn
        )
    )

    assert len(observed) == 6
    assert all(observed)
    assert len(set(observed)) == 1, "all chunks of one backfill must share one correlation id"
    assert get_correlation_id() is None


def test_ingest_backfill_chunked_inherits_caller_correlation_id(
    service: IngestionService,
) -> None:
    """When the caller already established a correlation id, every chunk must
    reuse it rather than minting new, unrelated ids per chunk."""
    observed: list[str | None] = []

    def process_fn(ledger_id: int, payload: object) -> None:
        observed.append(get_correlation_id())

    with CorrelationId("caller-backfill-id"):
        list(
            service.ingest_backfill_chunked(
                start_ledger=1, end_ledger=6, chunk_size=2, process_fn=process_fn
            )
        )
        assert get_correlation_id() == "caller-backfill-id"

    assert observed == ["caller-backfill-id"] * 6
    assert get_correlation_id() is None


def test_ingest_backfill_chunked_distinct_runs_get_distinct_ids(
    service: IngestionService,
) -> None:
    """Two independent, uninherited backfill calls must not share an id."""
    first_ids: list[str | None] = []
    second_ids: list[str | None] = []

    list(
        service.ingest_backfill_chunked(
            start_ledger=1,
            end_ledger=2,
            chunk_size=1,
            process_fn=lambda lid, p: first_ids.append(get_correlation_id()),
        )
    )
    list(
        service.ingest_backfill_chunked(
            start_ledger=3,
            end_ledger=4,
            chunk_size=1,
            process_fn=lambda lid, p: second_ids.append(get_correlation_id()),
        )
    )

    assert set(first_ids) != set(second_ids)
