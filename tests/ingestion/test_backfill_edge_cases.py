"""Edge-case coverage for ingestion backfills (issue #706).

Exercises the Polars preprocessing path
(``astroml.preprocessing.ledger_backfill``) plus the idempotent ledger store
(``upsert_processed_ledger``) against the cases that break backfills in
production:

* empty ledger ranges — no rows, no store writes, no crash;
* gaps between ledgers — present ledgers process, missing ones stay absent
  (no phantom rows);
* duplicate operations — ``(transaction_hash, operation_id)`` pairs collapse
  to one row so replays never double-count;
* restart mid-backfill — re-running over completed ledgers changes nothing;
* upsert idempotency — processing → completed → failed transitions update one
  row instead of inserting duplicates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from astroml.db.schema import ProcessedLedger
from astroml.preprocessing.ledger_backfill import (
    preprocess_ledger_backfill,
    scan_backfill_dataset,
    upsert_processed_ledger,
)


def _row(ledger: int, tx_hash: str, op_id: int, sender: str = "GAAA", **overrides):
    row = {
        "ledger_sequence": ledger,
        "source_account": sender,
        "transaction_hash": tx_hash,
        "created_at": "2024-01-01T00:00:00Z",
        "id": op_id,
        "type": "payment",
    }
    row.update(overrides)
    return row


@pytest.fixture()
def dataset_file(tmp_path):
    """Write rows as ndjson, return a writer + path pair."""

    def _write(rows) -> Path:
        path = tmp_path / "backfill.ndjson"
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return path

    return _write


@pytest.fixture()
def store():
    """Isolated SQLite store holding only the processed_ledgers table."""
    engine = create_engine("sqlite:///:memory:")
    ProcessedLedger.__table__.create(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()
    engine.dispose()


def _processed_frame(path: Path):
    frame = preprocess_ledger_backfill(scan_backfill_dataset(path, input_format="ndjson"))
    return frame.collect()


def _store_sequences(session: Session) -> dict[int, str]:
    rows = session.execute(select(ProcessedLedger)).scalars().all()
    return {row.ledger_sequence: row.status for row in rows}


class TestBackfillEdgeCases:
    def test_empty_ledger_range_writes_nothing(self, dataset_file, store: Session):
        out = _processed_frame(dataset_file([]))
        assert out.is_empty()
        assert _store_sequences(store) == {}

    def test_gap_between_ledgers_processes_present_only(self, dataset_file, store: Session):
        out = _processed_frame(
            dataset_file(
                [
                    _row(1, "h1", 1),
                    _row(2, "h2", 2),
                    _row(5, "h3", 3),  # ledgers 3-4 missing upstream
                ]
            )
        )
        assert sorted(out["ledger_sequence"].to_list()) == [1, 2, 5]
        for seq in (1, 2, 5):
            upsert_processed_ledger(store, seq, "test", "completed")
        assert _store_sequences(store) == {1: "completed", 2: "completed", 5: "completed"}

    def test_duplicate_operations_collapse_to_one_row(self, dataset_file):
        out = _processed_frame(
            dataset_file(
                [
                    _row(1, "h1", 1),
                    _row(1, "h1", 1),  # exact replay duplicate
                    _row(1, "h1", 2),  # same tx, distinct operation
                ]
            )
        )
        assert len(out) == 2
        assert sorted(out["operation_id"].to_list()) == [1, 2]

    def test_malformed_rows_are_dropped_not_fatal(self, dataset_file):
        out = _processed_frame(
            dataset_file(
                [
                    _row(1, "h1", 1),
                    {"source_account": "GB", "transaction_hash": "hX", "id": 9},  # no ledger/ts
                    {"ledger_sequence": 2, "transaction_hash": "hY", "id": 10},  # no sender/ts
                    {"ledger_sequence": 3, "source_account": "GC", "id": 11},  # no hash/ts
                ]
            )
        )
        assert out["ledger_sequence"].to_list() == [1]

    def test_restart_mid_backfill_is_idempotent(self, dataset_file, store: Session):
        rows = [_row(seq, f"h{seq}", seq) for seq in (1, 2, 3)]
        path = dataset_file(rows)

        # First run: process everything, mark completed.
        first = _processed_frame(path)
        assert len(first) == 3
        for seq in (1, 2, 3):
            upsert_processed_ledger(store, seq, "test", "completed")

        # Restart: re-running the same dataset yields identical output and
        # re-upserting touches the same three rows — no duplicates.
        second = _processed_frame(path)
        assert second.equals(first)
        for seq in (1, 2, 3):
            upsert_processed_ledger(store, seq, "test", "completed")
        stored = store.execute(select(ProcessedLedger)).scalars().all()
        assert len(stored) == 3
        assert _store_sequences(store) == {1: "completed", 2: "completed", 3: "completed"}

    def test_upsert_transitions_update_one_row(self, store: Session):
        upsert_processed_ledger(store, 9, "test", "processing")
        upsert_processed_ledger(
            store, 9, "test", "failed", error_message="horizon timeout"
        )
        upsert_processed_ledger(store, 9, "test", "completed", num_operations=4)

        rows = store.execute(
            select(ProcessedLedger).where(ProcessedLedger.ledger_sequence == 9)
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].status == "completed"
        assert rows[0].num_operations == 4

    def test_failed_ledger_can_be_retried_to_completed(self, store: Session):
        upsert_processed_ledger(store, 11, "test", "failed", error_message="boom")
        assert _store_sequences(store) == {11: "failed"}

        record = upsert_processed_ledger(store, 11, "test", "completed", num_operations=2)
        assert record.status == "completed"
        assert record.error_message is None
        assert _store_sequences(store) == {11: "completed"}
