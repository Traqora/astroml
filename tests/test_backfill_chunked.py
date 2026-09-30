"""Tests for chunked backfill memory optimisation — issue #766."""

from __future__ import annotations

import json
import os

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


class TestBackfillCheckpoint:
    def test_checkpoint_disabled_by_default(self, tmp_path):
        """Without resume_from_checkpoint, no checkpoint file is created."""
        svc = IngestionService()
        checkpoint_path = str(tmp_path / "checkpoint.json")

        list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=10,
                chunk_size=5,
                checkpoint_path=checkpoint_path,
            )
        )

        assert not os.path.exists(checkpoint_path)

    def test_checkpoint_saves_after_successful_chunk(self, tmp_path):
        """Checkpoint is saved after each successful chunk."""
        svc = IngestionService()
        checkpoint_path = str(tmp_path / "checkpoint.json")

        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=10,
                chunk_size=5,
                resume_from_checkpoint=True,
                checkpoint_path=checkpoint_path,
            )
        )

        assert os.path.exists(checkpoint_path)
        with open(checkpoint_path) as f:
            data = json.load(f)
        assert data["last_ledger"] == 10  # Last successfully processed

    def test_resume_from_checkpoint_skips_processed_chunks(self, tmp_path):
        """Resuming from checkpoint starts after the last saved position."""
        checkpoint_path = str(tmp_path / "checkpoint.json")
        processed_first_run: list[int] = []

        def process_fn(ledger_id: int, _payload):
            processed_first_run.append(ledger_id)

        # First run: process ledgers 1-10, interrupted after first chunk
        svc = IngestionService()
        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=20,
                chunk_size=5,
                process_fn=process_fn,
                resume_from_checkpoint=True,
                checkpoint_path=checkpoint_path,
            )
        )
        assert processed_first_run == list(range(1, 21))

        # Manually set checkpoint to ledger 5 to simulate crash mid-run
        with open(checkpoint_path, "w") as f:
            json.dump({"last_ledger": 5, "updated_at": "2024-01-01T00:00:00"}, f)

        # Second run: should resume from ledger 6
        processed_second_run: list[int] = []
        svc2 = IngestionService()
        chunks2 = list(
            svc2.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=20,
                chunk_size=5,
                process_fn=lambda lid, _: processed_second_run.append(lid),
                resume_from_checkpoint=True,
                checkpoint_path=checkpoint_path,
            )
        )

        # Should process 6-20 (skipped 1-5 from checkpoint)
        assert processed_second_run == list(range(6, 21))

    def test_resume_handles_missing_checkpoint(self, tmp_path):
        """When checkpoint file doesn't exist, starts from start_ledger."""
        checkpoint_path = str(tmp_path / "nonexistent.json")
        processed: list[int] = []

        svc = IngestionService()
        list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=5,
                chunk_size=2,
                process_fn=lambda lid, _: processed.append(lid),
                resume_from_checkpoint=True,
                checkpoint_path=checkpoint_path,
            )
        )

        assert processed == list(range(1, 6))

    def test_checkpoint_cleared_on_completion(self, tmp_path):
        """Checkpoint file is removed when backfill completes successfully."""
        checkpoint_path = str(tmp_path / "checkpoint.json")

        svc = IngestionService()
        list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=5,
                chunk_size=2,
                resume_from_checkpoint=True,
                checkpoint_path=checkpoint_path,
            )
        )

        assert not os.path.exists(checkpoint_path)

    def test_checkpoint_beyond_end_ledger_returns_early(self, tmp_path):
        """If checkpoint is already at or beyond end_ledger, nothing is processed."""
        checkpoint_path = str(tmp_path / "checkpoint.json")

        # Create a checkpoint beyond the target range
        with open(checkpoint_path, "w") as f:
            json.dump({"last_ledger": 100, "updated_at": "2024-01-01T00:00:00"}, f)

        processed: list[int] = []
        svc = IngestionService()
        chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=50,
                chunk_size=10,
                process_fn=lambda lid, _: processed.append(lid),
                resume_from_checkpoint=True,
                checkpoint_path=checkpoint_path,
            )
        )

        assert chunks == []
        assert processed == []

    def test_checkpoint_corrupted_file_is_ignored(self, tmp_path):
        """Corrupted checkpoint file is treated as missing, starts from scratch."""
        checkpoint_path = str(tmp_path / "corrupt.json")

        # Write invalid JSON
        with open(checkpoint_path, "w") as f:
            f.write("not valid json")

        processed: list[int] = []
        svc = IngestionService()
        list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=5,
                chunk_size=2,
                process_fn=lambda lid, _: processed.append(lid),
                resume_from_checkpoint=True,
                checkpoint_path=checkpoint_path,
            )
        )

        # Should process the full range (checkpoint ignored)
        assert processed == list(range(1, 6))
