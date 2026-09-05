"""Tests for chunked backfill memory optimisation — issue #766."""

from __future__ import annotations

import pytest

from astroml.ingestion.service import IngestionService, LedgerOutcome


class TestIngestBackfillChunked:
    def _make_service(self) -> IngestionService:
        return IngestionService()

    def test_processes_all_ledgers(self):
        svc = self._make_service()
        seen: list[int] = []

        def process(ledger_id: int, payload: object) -> None:
            seen.append(ledger_id)

        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=25,
                chunk_size=10,
                process_fn=process,
            )
        )
        assert sorted(seen) == list(range(1, 26))

    def test_yields_correct_chunk_count(self):
        svc = self._make_service()
        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=0,
                end_ledger=99,
                chunk_size=25,
            )
        )
        # 100 ledgers / chunk_size=25 → 4 chunks
        assert len(chunks) == 4

    def test_chunk_boundaries(self):
        svc = self._make_service()
        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=0,
                end_ledger=9,
                chunk_size=3,
            )
        )
        assert chunks[0]["chunk_start"] == 0
        assert chunks[0]["chunk_end"] == 2
        assert chunks[-1]["chunk_end"] == 9

    def test_processed_count_in_chunk(self):
        svc = self._make_service()
        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=5,
                chunk_size=10,
            )
        )
        assert len(chunks) == 1
        assert chunks[0]["processed"] == 5
        assert chunks[0]["skipped"] == 0
        assert chunks[0]["errors"] == 0

    def test_skipped_already_processed(self):
        svc = self._make_service()
        # Pre-process some ledgers
        list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=3,
                chunk_size=10,
            )
        )
        # Re-run over the same range — all should be skipped
        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=3,
                chunk_size=10,
            )
        )
        assert chunks[0]["skipped"] == 3
        assert chunks[0]["processed"] == 0

    def test_invalid_range_raises(self):
        svc = self._make_service()
        with pytest.raises(ValueError, match="end_ledger"):
            list(
                svc.ingest_backfill_chunked(
                    start_ledger=10,
                    end_ledger=5,
                    chunk_size=5,
                )
            )

    def test_invalid_chunk_size_raises(self):
        svc = self._make_service()
        with pytest.raises(ValueError, match="chunk_size"):
            list(
                svc.ingest_backfill_chunked(
                    start_ledger=0,
                    end_ledger=10,
                    chunk_size=0,
                )
            )

    def test_single_ledger_range(self):
        svc = self._make_service()
        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=42,
                end_ledger=42,
                chunk_size=10,
            )
        )
        assert len(chunks) == 1
        assert chunks[0]["processed"] == 1

    def test_chunk_larger_than_range(self):
        svc = self._make_service()
        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=0,
                end_ledger=4,
                chunk_size=100,
            )
        )
        assert len(chunks) == 1
        assert chunks[0]["processed"] == 5


class TestChunkedBenchmark:
    def test_run_chunked_benchmark_returns_result(self, tmp_path):
        from astroml.ingestion.benchmark import ChunkedBenchmarkResult, run_chunked_benchmark

        svc = IngestionService()
        result = run_chunked_benchmark(
            svc,
            start_ledger=0,
            end_ledger=49,
            chunk_size=10,
            results_path=str(tmp_path / "bench.jsonl"),
        )
        assert isinstance(result, ChunkedBenchmarkResult)
        assert result.total_processed == 50
        assert result.n_chunks == 5
        assert result.tx_per_sec > 0

    def test_benchmark_appends_jsonl(self, tmp_path):
        import json

        from astroml.ingestion.benchmark import run_chunked_benchmark

        out = tmp_path / "bench.jsonl"
        svc = IngestionService()
        run_chunked_benchmark(svc, 0, 9, chunk_size=5, results_path=str(out))
        run_chunked_benchmark(svc, 10, 19, chunk_size=5, results_path=str(out))

        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "chunk_size" in data
