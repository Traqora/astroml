"""Tests for database repository pattern and unit of work (issue #571)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from astroml.db.models import Account, Base, Ledger, ProcessedLedger, Transaction
from astroml.db.repositories import AccountRepository, LedgerRepository, ProcessedLedgerRepository, TransactionRepository
from astroml.db.unit_of_work import UnitOfWork, unit_of_work


@pytest.fixture(scope="function")
def db_session(tmp_path):
    db_file = tmp_path / "test_repos.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()
    engine.dispose()


class TestLedgerRepository:
    def test_save_and_get_by_sequence(self, db_session):
        repo = LedgerRepository(db_session)
        ledger = Ledger(sequence=1, hash="abc" * 21, closed_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        saved = repo.save(ledger)
        assert saved.sequence == 1
        found = repo.get_by_sequence(1)
        assert found is not None
        assert found.hash == "abc" * 21

    def test_get_by_range(self, db_session):
        repo = LedgerRepository(db_session)
        for i in range(1, 6):
            repo.save(Ledger(sequence=i, hash=f"hash{i:064d}", closed_at=datetime(2024, 1, i, tzinfo=timezone.utc)))
        ledgers = repo.get_by_range(2, 4)
        assert len(ledgers) == 3
        assert [l.sequence for l in ledgers] == [2, 3, 4]

    def test_get_latest_sequence(self, db_session):
        repo = LedgerRepository(db_session)
        assert repo.get_latest_sequence() is None
        for i in range(1, 4):
            repo.save(Ledger(sequence=i, hash=f"hash{i:064d}", closed_at=datetime(2024, 1, i, tzinfo=timezone.utc)))
        assert repo.get_latest_sequence() == 3

    def test_count(self, db_session):
        repo = LedgerRepository(db_session)
        assert repo.count() == 0
        for i in range(1, 4):
            repo.save(Ledger(sequence=i, hash=f"hash{i:064d}", closed_at=datetime(2024, 1, i, tzinfo=timezone.utc)))
        assert repo.count() == 3

    def test_delete(self, db_session):
        repo = LedgerRepository(db_session)
        ledger = Ledger(sequence=1, hash="abc" * 21, closed_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        repo.save(ledger)
        assert repo.count() == 1
        repo.delete(ledger)
        db_session.flush()
        assert repo.count() == 0


class TestTransactionRepository:
    def test_save_and_get_by_hash(self, db_session):
        repo = TransactionRepository(db_session)
        tx = Transaction(hash="txhash123", ledger_sequence=1, source_account="GABCDEF123", created_at=datetime(2024, 1, 1, tzinfo=timezone.utc), fee=100, operation_count=1, successful=True)
        saved = repo.save(tx)
        assert saved.hash == "txhash123"
        found = repo.get_by_hash("txhash123")
        assert found is not None
        assert found.source_account == "GABCDEF123"

    def test_get_by_account(self, db_session):
        repo = TransactionRepository(db_session)
        for i in range(3):
            repo.save(Transaction(hash=f"tx{i}", ledger_sequence=i, source_account="GABC", created_at=datetime(2024, 1, i + 1, tzinfo=timezone.utc), fee=100, operation_count=1, successful=True))
        results = repo.get_by_account("GABC")
        assert len(results) == 3

    def test_count_by_ledger(self, db_session):
        repo = TransactionRepository(db_session)
        for i in range(3):
            repo.save(Transaction(hash=f"tx{i}", ledger_sequence=5, source_account="GABC", created_at=datetime(2024, 1, i + 1, tzinfo=timezone.utc), fee=100, operation_count=1, successful=True))
        assert repo.count_by_ledger(5) == 3


class TestAccountRepository:
    def test_save_and_get(self, db_session):
        repo = AccountRepository(db_session)
        account = Account(account_id="GABCDEF123", balance=1000.0)
        saved = repo.save(account)
        assert saved.account_id == "GABCDEF123"
        found = repo.get_by_account_id("GABCDEF123")
        assert found is not None
        assert found.balance == 1000.0

    def test_get_active_since(self, db_session):
        repo = AccountRepository(db_session)
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)
        repo.save(Account(account_id="G1", updated_at=datetime(2024, 5, 1, tzinfo=timezone.utc)))
        repo.save(Account(account_id="G2", updated_at=datetime(2024, 7, 1, tzinfo=timezone.utc)))
        active = repo.get_active_since(datetime(2024, 6, 1, tzinfo=timezone.utc))
        assert len(active) == 1
        assert active[0].account_id == "G2"


class TestProcessedLedgerRepository:
    def test_save_and_check(self, db_session):
        repo = ProcessedLedgerRepository(db_session)
        pl = ProcessedLedger(ledger_sequence=100, source="test", status="completed")
        repo.save(pl)
        assert repo.is_processed(100) is True
        assert repo.is_processed(101) is False

    def test_get_by_status(self, db_session):
        repo = ProcessedLedgerRepository(db_session)
        for i in range(3):
            repo.save(ProcessedLedger(ledger_sequence=i, source="test", status="completed" if i < 2 else "failed"))
        completed = repo.get_by_status("completed")
        assert len(completed) == 2
        failed = repo.get_by_status("failed")
        assert len(failed) == 1


class TestUnitOfWork:
    def test_uow_provides_repositories(self, db_session):
        uow = UnitOfWork(db_session)
        assert uow.ledgers is not None
        assert uow.transactions is not None
        assert uow.accounts is not None
        assert uow.processed_ledgers is not None

    def test_uow_context_manager_commits(self, db_session):
        with UnitOfWork(db_session) as uow:
            uow.ledgers.save(Ledger(sequence=1, hash="abc" * 21, closed_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))
            uow.commit()
        repo = LedgerRepository(db_session)
        assert repo.count() == 1

    def test_uow_context_manager_rollback_on_error(self, db_session):
        try:
            with UnitOfWork(db_session) as uow:
                uow.ledgers.save(Ledger(sequence=2, hash="def" * 21, closed_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))
                raise ValueError("test error")
        except ValueError:
            pass
        repo = LedgerRepository(db_session)
        assert repo.count() == 0

    def test_unit_of_work_context_manager(self, db_session):
        with unit_of_work(db_session) as uow:
            uow.ledgers.save(Ledger(sequence=1, hash="abc" * 21, closed_at=datetime(2024, 1, 1, tzinfo=timezone.utc)))
        repo = LedgerRepository(db_session)
        assert repo.count() == 1
