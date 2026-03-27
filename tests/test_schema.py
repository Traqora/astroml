"""Tests for astroml.db.schema ORM models.

Uses SQLite in-memory for DDL smoke tests — no PostgreSQL required.
PostgreSQL-specific features (JSONB, partial indexes) are validated
structurally via metadata introspection rather than DDL execution.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from astroml.db.schema import (
    Account,
    Asset,
    Base,
    GraphAccount,
    GraphClaimDetail,
    GraphEdge,
    GraphPaymentDetail,
    GraphTransactionDetail,
    Ledger,
    Operation,
    Transaction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = create_engine("sqlite://")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine):
    """SQLAlchemy session bound to the in-memory engine."""
    with Session(engine) as s:
        yield s


# ---------------------------------------------------------------------------
# Import & table creation
# ---------------------------------------------------------------------------

def test_models_importable():
    """All five model classes import cleanly."""
    for cls in (
        Ledger,
        Transaction,
        Operation,
        Account,
        Asset,
        GraphAccount,
        GraphEdge,
        GraphTransactionDetail,
        GraphClaimDetail,
        GraphPaymentDetail,
    ):
        assert hasattr(cls, "__tablename__")


def test_create_all_tables(engine):
    """metadata.create_all() succeeds and produces the expected tables."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    assert table_names == {
        "accounts",
        "assets",
        "effects",
        "graph_accounts",
        "graph_claim_details",
        "graph_edges",
        "graph_payment_details",
        "graph_transaction_details",
        "ledgers",
        "normalized_transactions",
        "operations",
        "transactions",
    }


def test_table_names():
    """Each model maps to the correct table name."""
    assert Ledger.__tablename__ == "ledgers"
    assert Transaction.__tablename__ == "transactions"
    assert Operation.__tablename__ == "operations"
    assert Account.__tablename__ == "accounts"
    assert Asset.__tablename__ == "assets"
    assert GraphAccount.__tablename__ == "graph_accounts"
    assert GraphEdge.__tablename__ == "graph_edges"


# ---------------------------------------------------------------------------
# Column verification
# ---------------------------------------------------------------------------

def test_ledger_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("ledgers")}
    expected = {
        "sequence", "hash", "prev_hash", "closed_at",
        "successful_transaction_count", "failed_transaction_count",
        "operation_count", "total_coins", "fee_pool",
        "base_fee_in_stroops", "protocol_version",
    }
    assert expected <= cols


def test_transaction_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("transactions")}
    expected = {
        "hash", "ledger_sequence", "source_account", "created_at",
        "fee", "operation_count", "successful", "memo_type", "memo",
    }
    assert expected <= cols

    # FK to ledgers
    fks = inspector.get_foreign_keys("transactions")
    assert any(
        fk["referred_table"] == "ledgers"
        and fk["referred_columns"] == ["sequence"]
        for fk in fks
    )


def test_operation_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("operations")}
    expected = {
        "id", "transaction_hash", "application_order", "type",
        "source_account", "destination_account", "amount",
        "asset_code", "asset_issuer", "created_at", "details",
    }
    assert expected <= cols

    # FK to transactions
    fks = inspector.get_foreign_keys("operations")
    assert any(
        fk["referred_table"] == "transactions"
        and fk["referred_columns"] == ["hash"]
        for fk in fks
    )


def test_account_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("accounts")}
    expected = {
        "account_id", "balance", "sequence", "home_domain",
        "flags", "last_modified_ledger", "created_at", "updated_at",
    }
    assert expected <= cols


def test_asset_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("assets")}
    expected = {"id", "asset_type", "asset_code", "asset_issuer", "first_seen_ledger"}
    assert expected <= cols


def test_graph_account_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("graph_accounts")}
    expected = {
        "id",
        "account_address",
        "account_type",
        "first_seen_at",
        "last_seen_at",
        "created_at",
        "updated_at",
    }
    assert expected <= cols


def test_graph_edge_columns(engine):
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("graph_edges")}
    expected = {
        "id",
        "edge_type",
        "source_account_id",
        "destination_account_id",
        "asset_id",
        "occurred_at",
        "ledger_sequence",
        "event_index",
        "transaction_hash",
        "external_event_id",
        "amount",
        "status",
        "created_at",
    }
    assert expected <= cols

    fks = inspector.get_foreign_keys("graph_edges")
    assert any(
        fk["referred_table"] == "graph_accounts"
        and fk["referred_columns"] == ["id"]
        for fk in fks
    )
    assert any(
        fk["referred_table"] == "assets"
        and fk["referred_columns"] == ["id"]
        for fk in fks
    )


def test_graph_detail_columns(engine):
    inspector = inspect(engine)
    transaction_cols = {c["name"] for c in inspector.get_columns("graph_transaction_details")}
    claim_cols = {c["name"] for c in inspector.get_columns("graph_claim_details")}
    payment_cols = {c["name"] for c in inspector.get_columns("graph_payment_details")}

    assert {"edge_id", "edge_type", "successful", "operation_count", "fee", "memo_type", "memo", "details"} <= transaction_cols
    assert {"edge_id", "edge_type", "claim_reference", "claim_status", "expires_at", "details"} <= claim_cols
    assert {"edge_id", "edge_type", "payment_reference", "payment_status", "fee_amount", "settled_at", "details"} <= payment_cols


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

def test_relationships(session):
    """Ledger.transactions and Transaction.operations resolve correctly."""
    now = datetime.now(timezone.utc)

    ledger = Ledger(sequence=1, hash="a" * 64, closed_at=now)
    tx = Transaction(
        hash="b" * 64,
        ledger_sequence=1,
        source_account="G" + "A" * 55,
        created_at=now,
        fee=100,
        operation_count=1,
        successful=True,
    )
    op = Operation(
        id=1,
        transaction_hash="b" * 64,
        application_order=1,
        type="payment",
        source_account="G" + "A" * 55,
        destination_account="G" + "B" * 55,
        amount=50.0,
        asset_code="XLM",
        created_at=now,
    )

    session.add_all([ledger, tx, op])
    session.flush()

    assert tx in ledger.transactions
    assert op in tx.operations
    assert op.transaction is tx
    assert tx.ledger is ledger


def test_graph_relationships(session):
    """Graph edges connect graph accounts and subtype detail rows 1:1."""
    now = datetime.now(timezone.utc)

    asset = Asset(asset_type="native", asset_code="XLM")
    source = GraphAccount(
        account_address="G" + "S" * 55,
        account_type="wallet",
        first_seen_at=now,
        last_seen_at=now,
    )
    destination = GraphAccount(
        account_address="G" + "R" * 55,
        account_type="merchant",
        first_seen_at=now,
        last_seen_at=now,
    )

    session.add_all([asset, source, destination])
    session.flush()

    edge = GraphEdge(
        edge_type="payment",
        source_account_id=source.id,
        destination_account_id=destination.id,
        asset_id=asset.id,
        occurred_at=now,
        ledger_sequence=123,
        event_index=1,
        transaction_hash="e" * 64,
        external_event_id="payment:e" + "1" * 16,
        amount=25,
        status="settled",
    )
    session.add(edge)
    session.flush()

    detail = GraphPaymentDetail(
        edge_id=edge.id,
        payment_reference="invoice-42",
        payment_status="settled",
        fee_amount=1.5,
    )
    session.add(detail)
    session.flush()

    session.refresh(edge)

    assert edge.source_account is source
    assert edge.destination_account is destination
    assert edge.payment_detail is detail
    assert detail.edge is edge


# ---------------------------------------------------------------------------
# Round-trip insert & query
# ---------------------------------------------------------------------------

def test_insert_and_query(session):
    """Insert one row per table and read it back."""
    now = datetime.now(timezone.utc)

    ledger = Ledger(sequence=100, hash="c" * 64, closed_at=now)
    session.add(ledger)
    session.flush()

    tx = Transaction(
        hash="d" * 64,
        ledger_sequence=100,
        source_account="G" + "C" * 55,
        created_at=now,
        fee=200,
        operation_count=1,
        successful=True,
        memo_type="MEMO_TEXT",
        memo="test",
    )
    session.add(tx)
    session.flush()

    op = Operation(
        id=1,
        transaction_hash="d" * 64,
        application_order=1,
        type="create_account",
        source_account="G" + "C" * 55,
        destination_account="G" + "D" * 55,
        amount=100.0,
        created_at=now,
        details={"starting_balance": "100.0"},
    )
    session.add(op)

    account = Account(
        account_id="G" + "D" * 55,
        balance=100.0,
        sequence=1,
        created_at=now,
        updated_at=now,
    )
    session.add(account)

    asset = Asset(
        asset_type="native",
        asset_code="XLM",
    )
    session.add(asset)
    session.flush()

    # Query back
    assert session.get(Ledger, 100) is ledger
    assert session.get(Transaction, "d" * 64) is tx
    assert session.get(Operation, op.id) is op
    assert session.get(Account, "G" + "D" * 55) is account
    assert session.get(Asset, asset.id) is asset


def test_graph_edge_external_event_id_must_be_unique_per_type(session):
    """Composite uniqueness supports idempotent edge ingestion."""
    now = datetime.now(timezone.utc)

    source = GraphAccount(
        account_address="G" + "E" * 55,
        first_seen_at=now,
        last_seen_at=now,
    )
    destination = GraphAccount(
        account_address="G" + "F" * 55,
        first_seen_at=now,
        last_seen_at=now,
    )
    session.add_all([source, destination])
    session.flush()

    first = GraphEdge(
        edge_type="transaction",
        source_account_id=source.id,
        destination_account_id=destination.id,
        occurred_at=now,
        external_event_id="tx:duplicate-key",
    )
    duplicate = GraphEdge(
        edge_type="transaction",
        source_account_id=source.id,
        destination_account_id=destination.id,
        occurred_at=now,
        external_event_id="tx:duplicate-key",
    )

    session.add(first)
    session.flush()
    session.add(duplicate)

    with pytest.raises(IntegrityError):
        session.flush()


def test_graph_edge_detail_type_constraints_are_declared():
    """Subtype tables pin detail rows to a single graph edge type."""
    transaction_checks = {c.name for c in GraphTransactionDetail.__table__.constraints}
    claim_checks = {c.name for c in GraphClaimDetail.__table__.constraints}
    payment_checks = {c.name for c in GraphPaymentDetail.__table__.constraints}

    assert "ck_graph_transaction_details_edge_type" in transaction_checks
    assert "ck_graph_claim_details_edge_type" in claim_checks
    assert "ck_graph_payment_details_edge_type" in payment_checks


def test_graph_edge_indexes_cover_time_series_paths():
    """The graph mirror exposes the expected timeline-oriented indexes."""
    index_names = {index.name for index in GraphEdge.__table__.indexes}
    assert {
        "ix_graph_edges_occurred_at",
        "ix_graph_edges_source_occurred_at",
        "ix_graph_edges_destination_occurred_at",
        "ix_graph_edges_type_occurred_at",
        "ix_graph_edges_asset_occurred_at",
        "ix_graph_edges_status_occurred_at",
        "ix_graph_edges_tx_hash",
        "ix_graph_edges_ledger_event",
    } <= index_names
