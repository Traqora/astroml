"""Tests for database models (issue #571)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker


class TestModelImports:
    def test_import_from_models(self):
        from astroml.db.models import Base, Ledger, Transaction, Account
        assert Base is not None
        assert Ledger is not None
        assert Transaction is not None
        assert Account is not None

    def test_import_from_schema_backward_compat(self):
        from astroml.db.schema import Base, Ledger, Transaction, Account
        assert Base is not None
        assert Ledger is not None
        assert Transaction is not None
        assert Account is not None

    def test_import_from_db_init(self):
        from astroml.db import Base, Ledger, Transaction, Account
        assert Base is not None
        assert Ledger is not None
        assert Transaction is not None
        assert Account is not None

    def test_repositories_import(self):
        from astroml.db.repositories import AccountRepository, LedgerRepository, ProcessedLedgerRepository, TransactionRepository
        assert AccountRepository is not None
        assert LedgerRepository is not None
        assert TransactionRepository is not None
        assert ProcessedLedgerRepository is not None

    def test_unit_of_work_import(self):
        from astroml.db.unit_of_work import UnitOfWork, unit_of_work
        assert UnitOfWork is not None
        assert unit_of_work is not None


class TestTableCreation:
    def test_all_tables_created(self, tmp_path):
        from astroml.db.models import Base
        db_file = tmp_path / "test_models.db"
        engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)
        inspector = __import__("sqlalchemy", fromlist=["inspect"]).inspect(engine)
        tables = inspector.get_table_names()
        assert "ledgers" in tables
        assert "transactions" in tables
        assert "operations" in tables
        assert "accounts" in tables
        engine.dispose()


class TestModelInstances:
    def test_create_ledger(self, tmp_path):
        from astroml.db.models import Base, Ledger
        db_file = tmp_path / "test_ledger.db"
        engine = create_engine(f"sqlite:///{db_file}")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        ledger = Ledger(sequence=1, hash="a" * 64, closed_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        session.add(ledger)
        session.commit()
        result = session.query(Ledger).filter_by(sequence=1).first()
        assert result is not None
        assert result.hash == "a" * 64
        session.close()
        engine.dispose()
