"""
Shared pytest fixtures for API integration tests.
"""
from __future__ import annotations

import os

os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("DISABLE_SCHEDULER", "true")
os.environ.setdefault("DISABLE_WS_POLLER", "true")

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from astroml.db.schema import Base
import api.models.orm  # noqa: F401 — registers ORM models on Base.metadata
from api.models.orm import ApiAccount as Account, FraudAlert, LoyaltyPoints, ApiTransaction as Transaction, PointsTransaction

# ─── Engine / session ─────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_engine(tmp_path):
    """Ephemeral SQLite engine scoped to this test function."""
    db_file = tmp_path / "test_astroml.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_wal(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Session:
    """Clean session per test — all writes are rolled back on teardown."""
    SessionLocal = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


# ─── FastAPI TestClient with DB override ──────────────────────────────────────

@pytest.fixture(scope="function")
def client(db_engine, db_session):
    """FastAPI TestClient with DB dependencies replaced by the test session."""
    import os

    from api.app import app
    from api.database import get_db, get_sync_db, reset_engines
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    async_url = str(db_engine.url).replace("sqlite://", "sqlite+aiosqlite://")
    os.environ["DATABASE_URL"] = async_url
    reset_engines()

    async_engine = create_async_engine(
        async_url,
        connect_args={"check_same_thread": False},
    )

    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine, expire_on_commit=False, class_=AsyncSession
    )

    async def _override_async_db():
        async with AsyncSessionLocal() as session:
            yield session

    def _override_db():
        yield db_session

    app.dependency_overrides[get_sync_db] = _override_db
    app.dependency_overrides[get_db] = _override_async_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()
    async_engine.sync_engine.dispose()


# ─── ORM seed fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def seeded_account(db_session) -> Account:
    acc = Account(
        public_key="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
        first_seen=datetime(2024, 1, 1, tzinfo=timezone.utc),
        last_active=datetime(2024, 6, 1, tzinfo=timezone.utc),
        balance=1000.0,
    )
    db_session.add(acc)
    db_session.flush()
    return acc


@pytest.fixture()
def seeded_transaction(db_session) -> Transaction:
    tx = Transaction(
        hash="abc123def456abc123def456abc123def456abc123def456abc123def456ab12",
        ledger_sequence=100,
        source_account="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
        destination_account="GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
        amount=500.0,
        asset_code="XLM",
        fee=100,
        successful=True,
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    db_session.add(tx)
    db_session.flush()
    return tx


@pytest.fixture()
def seeded_alert(db_session) -> FraudAlert:
    alert = FraudAlert(
        account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
        pattern="sybil_cluster",
        risk_score=0.92,
        risk_level="high",
        description="Suspicious transaction velocity detected.",
        detected_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    db_session.add(alert)
    db_session.flush()
    return alert


@pytest.fixture()
def seeded_loyalty(db_session) -> LoyaltyPoints:
    lp = LoyaltyPoints(
        account_id="GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
        balance=2500,
        tier="silver",
        multiplier=1.1,
    )
    db_session.add(lp)
    db_session.flush()
    return lp


# ─── Raw dict fixtures (no DB, unit-level) ────────────────────────────────────

@pytest.fixture()
def sample_accounts():
    return [
        {"account_id": "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN", "sequence": 1},
        {"account_id": "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT", "sequence": 2},
        {"account_id": "GCKFBEIYV2U22IO2BJ4KVJOIP7XPWQGQFKKWXR6DOSJBV5SG3B3ORJF", "sequence": 3},
    ]


@pytest.fixture()
def sample_transactions():
    return [
        {
            "transaction_hash": "abc123",
            "ledger_sequence": 100,
            "source_account": "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            "fee_charged": 100,
            "operation_count": 1,
            "successful": True,
        },
        {
            "transaction_hash": "def456",
            "ledger_sequence": 101,
            "source_account": "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            "fee_charged": 200,
            "operation_count": 2,
            "successful": True,
        },
        {
            "transaction_hash": "ghi789",
            "ledger_sequence": 102,
            "source_account": "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            "fee_charged": 100,
            "operation_count": 1,
            "successful": False,
        },
    ]


@pytest.fixture()
def sample_alerts():
    return [
        {
            "alert_id": "a1",
            "account_id": "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN",
            "severity": "high",
            "resolved": False,
        },
        {
            "alert_id": "a2",
            "account_id": "GBZXN7PIRZGNMHGA7MUUUF4GWPY5AYPGZWXNBFNKKZ4YH67FQJG2FZT",
            "severity": "low",
            "resolved": True,
        },
    ]
