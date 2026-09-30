"""Incremental ingestion touches only new ledgers (issue #729).

A repeated run over a fixed range still walks every ledger in that range — each
already-processed id is looked up and yielded as ``skipped`` — so "catch me up"
costs grow with history rather than with how much is actually new.
``ingest_incremental`` asks the state store where it got to, asks the network
where the head is, and touches only what lies between.
"""

from __future__ import annotations

import pytest

from astroml.ingestion.service import IngestionService
from astroml.ingestion.state import StateStore


@pytest.fixture()
def store(tmp_path) -> StateStore:
    return StateStore(path=str(tmp_path / "state" / "ingestion_state.json"))


@pytest.fixture()
def service(store: StateStore) -> IngestionService:
    return IngestionService(state_store=store)


class _Recorder:
    """Records which ledgers were fetched and processed."""

    def __init__(self) -> None:
        self.fetched: list[int] = []
        self.processed: list[int] = []

    def fetch(self, ledger_id: int) -> dict:
        self.fetched.append(ledger_id)
        return {"ledger": ledger_id}

    def process(self, ledger_id: int, payload: object) -> None:
        self.processed.append(ledger_id)


class TestOnlyNewLedgers:
    def test_a_second_run_with_no_new_ledgers_does_no_work(self, service):
        rec = _Recorder()
        service.ingest_incremental(lambda: 105, rec.fetch, rec.process, start_from=100)
        first_pass = list(rec.fetched)

        rec.fetched.clear()
        result = service.ingest_incremental(lambda: 105, rec.fetch, rec.process)

        assert first_pass == [100, 101, 102, 103, 104, 105]
        assert rec.fetched == [], "an up-to-date run must not fetch anything"
        assert result.attempted == []

    def test_only_ledgers_past_the_watermark_are_fetched(self, service):
        rec = _Recorder()
        service.ingest_incremental(lambda: 102, rec.fetch, rec.process, start_from=100)

        rec.fetched.clear()
        service.ingest_incremental(lambda: 105, rec.fetch, rec.process)

        assert rec.fetched == [103, 104, 105]

    def test_old_ledgers_are_not_even_visited(self, service, store):
        rec = _Recorder()
        service.ingest_incremental(lambda: 1000, rec.fetch, rec.process, start_from=1000)

        result = service.ingest_incremental(lambda: 1002, rec.fetch, rec.process)

        # The whole point: the attempted set is proportional to what is new, not
        # to how much history the store has seen.
        assert result.attempted == [1001, 1002]

    def test_the_watermark_advances(self, service, store):
        rec = _Recorder()
        service.ingest_incremental(lambda: 10, rec.fetch, rec.process, start_from=8)

        assert store.load().last_processed_ledger == 10


class TestColdStart:
    def test_a_cold_store_starts_at_the_head_by_default(self, service):
        rec = _Recorder()

        service.ingest_incremental(lambda: 500, rec.fetch, rec.process)

        # Incremental mode is "start watching from now", not "backfill all of
        # history" — a backfill is what ingest(start, end) is for.
        assert rec.fetched == [500]

    def test_start_from_backfills_from_an_explicit_point(self, service):
        rec = _Recorder()

        service.ingest_incremental(lambda: 5, rec.fetch, rec.process, start_from=1)

        assert rec.fetched == [1, 2, 3, 4, 5]

    def test_start_from_is_ignored_once_state_exists(self, service):
        rec = _Recorder()
        service.ingest_incremental(lambda: 10, rec.fetch, rec.process, start_from=10)

        rec.fetched.clear()
        service.ingest_incremental(lambda: 12, rec.fetch, rec.process, start_from=1)

        assert rec.fetched == [11, 12], "state must win over start_from"


class TestUnknownHead:
    def test_an_unknown_head_does_nothing(self, service):
        rec = _Recorder()

        result = service.ingest_incremental(lambda: None, rec.fetch, rec.process)

        assert rec.fetched == []
        assert result.attempted == []
        assert result.errors == []

    def test_a_head_behind_the_watermark_does_nothing(self, service):
        rec = _Recorder()
        service.ingest_incremental(lambda: 100, rec.fetch, rec.process, start_from=100)

        rec.fetched.clear()
        # A reorg or a lagging replica reporting an older head must not rewind.
        result = service.ingest_incremental(lambda: 50, rec.fetch, rec.process)

        assert rec.fetched == []
        assert result.attempted == []


class TestBounding:
    def test_max_ledgers_caps_a_single_run(self, service):
        rec = _Recorder()

        service.ingest_incremental(
            lambda: 1000, rec.fetch, rec.process, start_from=1, max_ledgers=10
        )

        assert rec.fetched == list(range(1, 11))

    def test_the_remainder_is_picked_up_by_the_next_run(self, service):
        rec = _Recorder()
        service.ingest_incremental(
            lambda: 20, rec.fetch, rec.process, start_from=1, max_ledgers=10
        )

        rec.fetched.clear()
        service.ingest_incremental(lambda: 20, rec.fetch, rec.process, max_ledgers=10)

        assert rec.fetched == list(range(11, 21))

    def test_max_ledgers_must_be_positive(self, service):
        with pytest.raises(ValueError, match="max_ledgers"):
            service.ingest_incremental(lambda: 10, max_ledgers=0, start_from=1)


class TestIdempotencyAndRestartSafety:
    def test_reprocessing_an_already_seen_ledger_is_skipped_not_refetched(self, service, store):
        rec = _Recorder()
        service.ingest_incremental(lambda: 5, rec.fetch, rec.process, start_from=1)

        # Simulate a watermark that regressed (e.g. state restored from an older
        # backup) while the processed set is intact.
        state = store.load()
        state.last_processed_ledger = 2
        store.save(state)

        rec.fetched.clear()
        result = service.ingest_incremental(lambda: 5, rec.fetch, rec.process)

        assert rec.fetched == [], "already-processed ledgers must not be fetched again"
        assert result.skipped == [3, 4, 5]

    def test_state_survives_a_new_service_instance(self, store):
        rec = _Recorder()
        IngestionService(state_store=store).ingest_incremental(
            lambda: 7, rec.fetch, rec.process, start_from=5
        )

        rec.fetched.clear()
        # A fresh process reading the same state file continues where it left off.
        IngestionService(state_store=StateStore(path=store.path)).ingest_incremental(
            lambda: 9, rec.fetch, rec.process
        )

        assert rec.fetched == [8, 9]

    def test_each_ledger_is_processed_exactly_once_across_runs(self, service):
        rec = _Recorder()
        for head in (3, 6, 6, 9):
            service.ingest_incremental(lambda h=head: h, rec.fetch, rec.process, start_from=1)

        assert rec.processed == list(range(1, 10))
        assert len(rec.processed) == len(set(rec.processed))


class TestNoRegressionToExplicitRanges:
    def test_explicit_range_ingestion_still_works(self, service):
        rec = _Recorder()

        result = service.ingest(
            start_ledger=1, end_ledger=3, fetch_fn=rec.fetch, process_fn=rec.process
        )

        assert result.processed == [1, 2, 3]
        assert rec.fetched == [1, 2, 3]

    def test_incremental_and_explicit_runs_share_one_state(self, service):
        rec = _Recorder()
        service.ingest(start_ledger=1, end_ledger=3, fetch_fn=rec.fetch, process_fn=rec.process)

        rec.fetched.clear()
        service.ingest_incremental(lambda: 5, rec.fetch, rec.process)

        assert rec.fetched == [4, 5]
