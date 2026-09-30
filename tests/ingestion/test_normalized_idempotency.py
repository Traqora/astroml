"""Ingestion writes are idempotent on replay (issue #728).

``normalized_transactions`` carries only a surrogate primary key, and a freshly
normalized row has no ``id``, so ``session.merge()`` — which resolves by primary
key — inserted a duplicate every time the same activity was written twice.  A
replayed Horizon event, a resumed batch after a restart, or a refetched page all
produced duplicate activity rows, which the graph builder and feature store then
counted twice.

These tests key the writes on Horizon's own identity for the row — the ledger
and operation id, plus the hop index for the several rows one path-payment
operation decomposes into — and assert that writing the same activity twice
lands on one row.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from astroml.db.models import Base, NormalizedTransaction
from astroml.db.repositories import NormalizedTransactionRepository
from astroml.ingestion.batch import BatchBuffer
from astroml.ingestion.normalizer import (
    normalize_operation,
    normalize_path_payment_hops,
)
from astroml.ingestion.parsers import ledger_sequence_from_operation_id

# Horizon operation ids are toids: ledger << 32 | tx_order << 12 | op_index.
# This one decodes to ledger 12554 and is the id used throughout the fixtures.
OP_ID = 53919970611201
LEDGER = 12554


@pytest.fixture()
def session(tmp_path):
    """A session over a fresh SQLite database built from the ORM models."""
    db_file = tmp_path / "test_idempotency.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = factory()
    yield db
    db.rollback()
    db.close()
    engine.dispose()


@pytest.fixture()
def payment_payload():
    """A raw Horizon payment operation."""
    return {
        "id": str(OP_ID),
        "type": "payment",
        "source_account": "G_SENDER",
        "to": "G_RECEIVER",
        "amount": "100.5",
        "asset_type": "native",
        "created_at": "2024-01-15T12:30:00Z",
        "transaction_hash": "a" * 64,
    }


@pytest.fixture()
def path_payment_payload():
    """A raw Horizon path payment with two hops (XLM -> USDC -> XLM)."""
    return {
        "id": str(OP_ID),
        "type": "path_payment_strict_send",
        "source_account": "G_SENDER",
        "to": "G_RECEIVER",
        "source_asset_type": "native",
        "source_amount": "10.0",
        "asset_type": "credit_alphanum4",
        "asset_code": "USDC",
        "asset_issuer": "G_ISSUER",
        "destination_amount": "9.5",
        "path": [{"asset_type": "native"}],
        "created_at": "2024-01-15T12:30:00Z",
        "transaction_hash": "b" * 64,
    }


def _count(session) -> int:
    return session.execute(select(func.count()).select_from(NormalizedTransaction)).scalar_one()


class TestNormalizedKeyExtraction:
    """The natural key comes from data Horizon already sends."""

    def test_ledger_is_decoded_from_the_operation_id(self):
        """The ledger is the high 32 bits of the toid, so no extra field is needed."""
        assert ledger_sequence_from_operation_id(OP_ID) == LEDGER

    def test_normalize_operation_records_ledger_and_operation_id(self, payment_payload):
        record = normalize_operation(payment_payload)

        assert record.ledger_sequence == LEDGER
        assert record.operation_id == OP_ID
        assert record.hop_index == 0

    def test_payload_without_an_id_is_written_without_a_key(self):
        """A payload with no operation id is not keyed on a guess."""
        record = normalize_operation(
            {
                "type": "payment",
                "source_account": "G_SENDER",
                "to": "G_RECEIVER",
                "amount": "1.0",
                "asset_type": "native",
                "created_at": "2024-01-15T12:30:00Z",
                "transaction_hash": "c" * 64,
            }
        )

        assert record.ledger_sequence is None
        assert record.operation_id is None


class TestReplayDoesNotDuplicate:
    """Writing the same activity twice lands on one row."""

    def test_same_operation_upserted_twice_yields_one_row(self, session, payment_payload):
        repo = NormalizedTransactionRepository(session)

        repo.upsert(normalize_operation(payment_payload))
        session.commit()
        repo.upsert(normalize_operation(payment_payload))
        session.commit()

        assert _count(session) == 1

    def test_replay_survives_a_new_session(self, tmp_path, payment_payload):
        """A resumed ingestion process re-reading the same page does not duplicate.

        This is the restart case: the retry happens in a different session with
        no identity map left over from the first write.
        """
        db_file = tmp_path / "restart.db"
        engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

        first = factory()
        NormalizedTransactionRepository(first).upsert(normalize_operation(payment_payload))
        first.commit()
        first.close()

        second = factory()
        NormalizedTransactionRepository(second).upsert(normalize_operation(payment_payload))
        second.commit()
        count = _count(second)
        second.close()
        engine.dispose()

        assert count == 1

    def test_buffered_ingestion_replay_does_not_duplicate(self, session, payment_payload):
        """The chunked batch path is idempotent too.

        ``stream.py`` buffers operations and normalizes them through
        ``BatchBuffer``, flushing every ``chunk_size`` rows, so a replayed event
        re-enters this path rather than the single-row write.
        """
        buffer = BatchBuffer(session, chunk_size=1, flush_on_exit=True)
        buffer.add(normalize_operation(payment_payload))
        buffer.add(normalize_operation(payment_payload))

        assert _count(session) == 1

    def test_replay_updates_the_row_rather_than_ignoring_it(self, session, payment_payload):
        """A restated operation corrects the stored values in place."""
        repo = NormalizedTransactionRepository(session)
        repo.upsert(normalize_operation(payment_payload))
        session.commit()

        corrected = normalize_operation(payment_payload)
        corrected.amount = 250.0
        repo.upsert(corrected)
        session.commit()

        stored = session.execute(select(NormalizedTransaction)).scalar_one()
        assert _count(session) == 1
        assert float(stored.amount) == 250.0

    def test_distinct_operations_stay_distinct(self, session, payment_payload):
        repo = NormalizedTransactionRepository(session)

        repo.upsert(normalize_operation(payment_payload))
        other = dict(payment_payload, id=str(OP_ID + 1), transaction_hash="d" * 64)
        repo.upsert(normalize_operation(other))
        session.commit()

        assert _count(session) == 2

    def test_rows_without_a_key_are_still_written(self, session):
        """Unkeyable payloads are appended, not dropped."""
        repo = NormalizedTransactionRepository(session)
        payload = {
            "type": "payment",
            "source_account": "G_SENDER",
            "to": "G_RECEIVER",
            "amount": "1.0",
            "asset_type": "native",
            "created_at": "2024-01-15T12:30:00Z",
            "transaction_hash": "e" * 64,
        }

        repo.upsert(normalize_operation(payload))
        repo.upsert(normalize_operation(payload))
        session.commit()

        assert _count(session) == 2


class TestPathPaymentHops:
    """One path payment writes several rows, each idempotent on its own."""

    def test_each_hop_is_a_distinct_row(self, session, path_payment_payload):
        hops = normalize_path_payment_hops(path_payment_payload)

        assert len(hops) > 1
        assert [h.hop_index for h in hops] == list(range(len(hops)))

    def test_hops_share_the_operation_key_but_differ_by_index(self, path_payment_payload):
        hops = normalize_path_payment_hops(path_payment_payload)

        for hop in hops:
            assert hop.operation_id == OP_ID
            assert hop.ledger_sequence == LEDGER

    def test_hop_transaction_hash_fits_its_column(self, path_payment_payload):
        """Hops keep the real hash instead of a suffixed one that overflowed.

        The previous encoding appended ``_hopN`` to the transaction hash, which
        pushed the value past the column's 64 characters and would not store on
        Postgres.
        """
        hops = normalize_path_payment_hops(path_payment_payload)

        for hop in hops:
            assert hop.transaction_hash == "b" * 64
            assert len(hop.transaction_hash) <= 64

    def test_replaying_a_path_payment_writes_no_extra_rows(self, session, path_payment_payload):
        repo = NormalizedTransactionRepository(session)
        hops = normalize_path_payment_hops(path_payment_payload)

        repo.batch_upsert(hops)
        repo.batch_upsert(normalize_path_payment_hops(path_payment_payload))

        assert _count(session) == len(hops)

    def test_batch_upsert_resolves_keys_in_one_pass(self, session, path_payment_payload):
        repo = NormalizedTransactionRepository(session)
        hops = normalize_path_payment_hops(path_payment_payload)

        written = repo.batch_upsert(hops)

        assert written == len(hops)
        assert repo.count() == len(hops)


class TestDatabaseLevelGuarantee:
    """The key is enforced by the table, not only by the repository helper."""

    def test_duplicate_natural_key_is_rejected(self, session, payment_payload):
        session.add(normalize_operation(payment_payload))
        session.commit()

        session.add(normalize_operation(payment_payload))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        assert _count(session) == 1

    def test_same_operation_in_different_ledgers_is_allowed(self, session, payment_payload):
        """The ledger is part of the key, so ids are scoped by it."""
        session.add(normalize_operation(payment_payload))
        # Same operation id, keyed against a different ledger: a distinct row.
        replayed = normalize_operation(payment_payload)
        replayed.ledger_sequence = LEDGER + 1
        session.add(replayed)
        session.commit()

        assert _count(session) == 2


class TestTimestampPreserved:
    """Replays keep the original activity time rather than the retry time."""

    def test_timestamp_comes_from_the_payload(self, session, payment_payload):
        repo = NormalizedTransactionRepository(session)
        repo.upsert(normalize_operation(payment_payload))
        session.commit()

        stored = session.execute(select(NormalizedTransaction)).scalar_one()
        # Compared field-wise because SQLite returns naive datetimes.
        assert (stored.timestamp.year, stored.timestamp.month, stored.timestamp.day) == (
            2024,
            1,
            15,
        )
        assert (stored.timestamp.hour, stored.timestamp.minute) == (12, 30)
