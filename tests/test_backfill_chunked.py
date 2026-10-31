"""Tests for chunked backfill memory optimisation — issue #766."""

from __future__ import annotations

import json
import os

import pytest

from astroml.ingestion.service import IngestionService, LedgerOutcome
from astroml.ingestion.state import StateStore


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


class TestBackfillRerunIdempotency:
    """Idempotency test for backfill reruns (issue #829).

    `ingest_backfill_chunked`'s docstring and module-level docs both claim
    idempotent reruns ("state is updated per-ledger, ensuring safe
    retries"), but had no dedicated regression test for the chunked
    backfill path specifically (only `test_skipped_already_processed`
    above, which covers a single chunk). These exercise a full rerun
    across multiple chunks, with a real `process_fn` side effect, so a
    regression that reprocesses already-completed ledgers shows up as a
    duplicate call rather than only as a `processed`/`skipped` count.
    """

    def test_rerunning_the_full_range_processes_nothing_the_second_time(self, tmp_path):
        svc = IngestionService(state_store=StateStore(path=str(tmp_path / "state.json")))
        calls: list[int] = []

        def process(ledger_id: int, _payload) -> None:
            calls.append(ledger_id)

        first_chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=23,
                chunk_size=7,
                process_fn=process,
            )
        )
        assert calls == list(range(1, 24))
        assert sum(c["processed"] for c in first_chunks) == 23
        assert sum(c["skipped"] for c in first_chunks) == 0

        second_chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=23,
                chunk_size=7,
                process_fn=process,
            )
        )

        # No new calls to process_fn: every ledger in the rerun is already
        # recorded as processed, so `process` must not fire again for any
        # of them.
        assert calls == list(range(1, 24))
        assert sum(c["processed"] for c in second_chunks) == 0
        assert sum(c["skipped"] for c in second_chunks) == 23

    def test_rerunning_a_partially_overlapping_range_only_processes_new_ledgers(self, tmp_path):
        svc = IngestionService(state_store=StateStore(path=str(tmp_path / "state.json")))
        calls: list[int] = []

        def process(ledger_id: int, _payload) -> None:
            calls.append(ledger_id)

        list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=10,
                chunk_size=4,
                process_fn=process,
            )
        )
        assert calls == list(range(1, 11))

        # Overlaps 1-10 (already processed) and extends to 15 (new).
        second_chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=15,
                chunk_size=4,
                process_fn=process,
            )
        )

        assert calls == list(range(1, 16))
        assert sum(c["processed"] for c in second_chunks) == 5
        assert sum(c["skipped"] for c in second_chunks) == 10

    def test_rerun_after_a_mid_backfill_error_does_not_reprocess_completed_chunks(self, tmp_path):
        """A chunk that raises still lets the next `ingest_backfill_chunked`
        call pick back up without redoing the chunks that already
        succeeded, since each ledger's processed state was persisted
        per-chunk rather than only at the very end of the call."""
        svc = IngestionService(state_store=StateStore(path=str(tmp_path / "state.json")))
        calls: list[int] = []
        fail_on = {5}

        def flaky_process(ledger_id: int, _payload) -> None:
            if ledger_id in fail_on:
                raise RuntimeError(f"simulated failure at ledger {ledger_id}")
            calls.append(ledger_id)

        first_chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=12,
                chunk_size=4,
                process_fn=flaky_process,
            )
        )
        # Chunk 1-4 and 9-12 succeed; the chunk containing ledger 5 (5-8)
        # is reported as an error and its ledgers are never marked processed.
        assert calls == [1, 2, 3, 4, 9, 10, 11, 12]
        assert sum(c["errors"] for c in first_chunks) == 1

        fail_on.clear()
        second_chunks = list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=12,
                chunk_size=4,
                process_fn=flaky_process,
            )
        )

        # Only the previously-failed chunk's ledgers (5-8) are processed on
        # rerun; 1-4 and 9-12 are skipped, not reprocessed.
        assert calls == [1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8]
        assert sum(c["processed"] for c in second_chunks) == 4
        assert sum(c["skipped"] for c in second_chunks) == 8
        assert sum(c["errors"] for c in second_chunks) == 0


class TestBackfillCorrelationId:
    """One correlation id shared across every chunk of one backfill run
    (issue #957), restored alongside the checkpoint fix for issue #829:
    the two features had been merged together incorrectly, dropping this
    behavior (and the checkpoint/current-ledger initialization entirely)
    while leaving the docstring claiming it still happened."""

    def test_same_correlation_id_is_active_throughout_every_chunk(self, tmp_path):
        from astroml.utils.logging import get_correlation_id

        svc = IngestionService(state_store=StateStore(path=str(tmp_path / "state.json")))
        seen_ids: list[str | None] = []

        def process(_ledger_id: int, _payload) -> None:
            seen_ids.append(get_correlation_id())

        list(
            svc.ingest_backfill_chunked(
                start_ledger=1,
                end_ledger=20,
                chunk_size=5,
                process_fn=process,
            )
        )

        assert len(seen_ids) == 20
        assert len(set(seen_ids)) == 1
        assert seen_ids[0] is not None

    def test_correlation_id_inherits_an_already_active_scope(self, tmp_path):
        from astroml.utils.logging import CorrelationId, get_correlation_id

        svc = IngestionService(state_store=StateStore(path=str(tmp_path / "state.json")))
        seen_ids: list[str | None] = []

        def process(_ledger_id: int, _payload) -> None:
            seen_ids.append(get_correlation_id())

        with CorrelationId("caller-supplied-id"):
            list(
                svc.ingest_backfill_chunked(
                    start_ledger=1,
                    end_ledger=10,
                    chunk_size=5,
                    process_fn=process,
                )
            )

        assert all(cid == "caller-supplied-id" for cid in seen_ids)
