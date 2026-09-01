"""Tests for per-batch ingestion progress/throughput metrics (Issue #727)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY


@pytest.fixture(autouse=True)
def _reset_batch_metrics():
    """Reset batch metric counters before each test so tests are order-independent."""
    INGESTION_BATCH_LEDGERS.labels(status="processed")._value.set(0.0)
    INGESTION_BATCH_LEDGERS.labels(status="skipped")._value.set(0.0)
    INGESTION_BATCH_LEDGERS.labels(status="error")._value.set(0.0)
    INGESTION_BATCH_THROUGHPUT._value.set(0.0)
    INGESTION_BATCH_DURATION_SECONDS._sum.set(0.0)

from astroml.ingestion.batch_metrics import BatchMetricsRecorder
from astroml.ingestion.metrics import (
    INGESTION_BATCH_DURATION_SECONDS,
    INGESTION_BATCH_LEDGERS,
    INGESTION_BATCH_THROUGHPUT,
)


class _Outcome:
    def __init__(self, status: str) -> None:
        self.status = status


class TestBatchMetricsRecorder:
    def test_records_processed_and_skipped(self) -> None:
        recorder = BatchMetricsRecorder()
        recorder.start()
        recorder.observe(_Outcome("processed"))
        recorder.observe(_Outcome("processed"))
        recorder.observe(_Outcome("skipped"))
        recorder.finish()

        assert (
            INGESTION_BATCH_LEDGERS.labels(status="processed")._value.get() == 2.0
        )
        assert INGESTION_BATCH_LEDGERS.labels(status="skipped")._value.get() == 1.0

    def test_records_error_outcome(self) -> None:
        recorder = BatchMetricsRecorder()
        recorder.start()
        recorder.observe(_Outcome("error"))
        recorder.finish()

        assert INGESTION_BATCH_LEDGERS.labels(status="error")._value.get() == 1.0

    def test_throughput_is_non_negative(self) -> None:
        recorder = BatchMetricsRecorder()
        recorder.start()
        recorder.observe(_Outcome("processed"))
        recorder.finish()

        throughput = INGESTION_BATCH_THROUGHPUT._value.get()
        assert throughput >= 0.0

    def test_duration_observed(self) -> None:
        recorder = BatchMetricsRecorder()
        recorder.start()
        time.sleep(0.01)
        recorder.finish()

        # Histogram exposes samples through _sum and _count
        assert INGESTION_BATCH_DURATION_SECONDS._sum.get() >= 0.01

    def test_start_resets_counters(self) -> None:
        recorder = BatchMetricsRecorder()
        recorder.start()
        recorder.observe(_Outcome("processed"))
        recorder.finish()

        recorder.start()
        recorder.finish()

        # The second batch has no new processed observations, so the counter
        # should still be 1.0 (counters are monotonic).
        assert INGESTION_BATCH_LEDGERS.labels(status="processed")._value.get() == 1.0


class TestIngestionServiceBatchMetrics:
    def test_emits_batch_metrics_after_batch_boundary(self, tmp_path) -> None:
        from astroml.ingestion.service import IngestionService
        from astroml.ingestion.state import StateStore

        state_path = tmp_path / "state.json"
        store = StateStore(str(state_path))
        service = IngestionService(store)

        list(service.ingest_stream(start_ledger=1, end_ledger=5, batch_size=5))

        assert INGESTION_BATCH_LEDGERS.labels(status="processed")._value.get() == 5.0
        assert INGESTION_BATCH_THROUGHPUT._value.get() >= 0.0

    def test_emits_batch_metrics_for_partial_final_batch(self, tmp_path) -> None:
        from astroml.ingestion.service import IngestionService
        from astroml.ingestion.state import StateStore

        state_path = tmp_path / "state.json"
        store = StateStore(str(state_path))
        service = IngestionService(store)

        list(service.ingest_stream(start_ledger=1, end_ledger=3, batch_size=5))

        assert INGESTION_BATCH_LEDGERS.labels(status="processed")._value.get() == 3.0
