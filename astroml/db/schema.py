"""SQLAlchemy ORM models for AstroML storage.

The schema has two layers:

- Raw Stellar blockchain storage used by the current ingestion pipeline.
- A normalized graph-mirror layer for account-centric time-series retrieval.

Five raw tables model the core Stellar data needed for graph ML:

- **ledgers** — temporal anchor; one row per closed ledger (~5-6 s apart).
- **transactions** — one row per transaction, linked to a ledger.
- **operations** — one row per operation (the primary graph-edge table).
- **accounts** — latest known account state snapshots.
- **assets** — asset registry (unique by code + issuer).

Indexes follow the project requirement of ``account_id + timestamp``
composite indexes on both transactions and operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Declarative base for all AstroML models."""


# ---------------------------------------------------------------------------
# Ledgers
# ---------------------------------------------------------------------------

class Ledger(Base):
    """One row per closed Stellar ledger."""

    __tablename__ = "ledgers"

    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    prev_hash: Mapped[Optional[str]] = mapped_column(String(64))
    closed_at: Mapped[datetime] = mapped_column(nullable=False)
    successful_transaction_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    failed_transaction_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    operation_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    total_coins: Mapped[Optional[float]] = mapped_column(Numeric)
    fee_pool: Mapped[Optional[float]] = mapped_column(Numeric)
    base_fee_in_stroops: Mapped[Optional[int]] = mapped_column(Integer)
    protocol_version: Mapped[Optional[int]] = mapped_column(Integer)

    # Relationships
    transactions: Mapped[list[Transaction]] = relationship(
        back_populates="ledger",
    )

    __table_args__ = (
        Index("ix_ledgers_closed_at", "closed_at"),
    )


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

class Transaction(Base):
    """One row per Stellar transaction."""

    __tablename__ = "transactions"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    ledger_sequence: Mapped[int] = mapped_column(
        Integer, ForeignKey("ledgers.sequence"), nullable=False
    )
    source_account: Mapped[str] = mapped_column(String(56), nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    fee: Mapped[int] = mapped_column(BigInteger, nullable=False)
    operation_count: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    successful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    memo_type: Mapped[Optional[str]] = mapped_column(String(16))
    memo: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    ledger: Mapped[Ledger] = relationship(back_populates="transactions")
    operations: Mapped[list[Operation]] = relationship(
        back_populates="transaction",
    )

    __table_args__ = (
        Index("ix_transactions_source_account_created_at", "source_account", "created_at"),
        Index("ix_transactions_ledger_sequence", "ledger_sequence"),
    )


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

class Operation(Base):
    """One row per Stellar operation — the primary graph-edge table.

    Common columns (``source_account``, ``destination_account``, ``amount``,
    ``asset_code``, ``asset_issuer``) cover graph-relevant fields for the
    majority of operation types.  The ``details`` JSONB column stores
    type-specific fields for the remaining types.

    ``created_at`` is denormalized from the parent transaction to avoid a
    JOIN on every temporal range query.
    """

    __tablename__ = "operations"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    transaction_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("transactions.hash"), nullable=False
    )
    application_order: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_account: Mapped[str] = mapped_column(String(56), nullable=False)
    destination_account: Mapped[Optional[str]] = mapped_column(String(56))
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    asset_code: Mapped[Optional[str]] = mapped_column(String(12))
    asset_issuer: Mapped[Optional[str]] = mapped_column(String(56))
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )

    # Relationships
    transaction: Mapped[Transaction] = relationship(back_populates="operations")

    __table_args__ = (
        Index("ix_operations_source_created_at", "source_account", "created_at"),
        Index(
            "ix_operations_dest_created_at",
            "destination_account",
            "created_at",
            postgresql_where=(destination_account.isnot(None)),
        ),
        Index("ix_operations_transaction_hash", "transaction_hash"),
        Index("ix_operations_type", "type"),
    )


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

class Account(Base):
    """Latest known state of a Stellar account."""

    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String(56), primary_key=True)
    balance: Mapped[Optional[float]] = mapped_column(Numeric)
    sequence: Mapped[Optional[int]] = mapped_column(BigInteger)
    home_domain: Mapped[Optional[str]] = mapped_column(String(32))
    flags: Mapped[int] = mapped_column(Integer, server_default="0")
    last_modified_ledger: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[Optional[datetime]] = mapped_column()
    updated_at: Mapped[Optional[datetime]] = mapped_column()

    __table_args__ = (
        Index("ix_accounts_updated_at", "updated_at"),
    )


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

class Asset(Base):
    """Asset registry — unique by (code, issuer).

    Native XLM has ``asset_issuer = NULL``.  The unique constraint uses
    ``COALESCE(asset_issuer, '')`` to handle NULL correctly.
    """

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_type: Mapped[str] = mapped_column(String(16), nullable=False)
    asset_code: Mapped[str] = mapped_column(String(12), nullable=False)
    asset_issuer: Mapped[Optional[str]] = mapped_column(String(56))
    first_seen_ledger: Mapped[Optional[int]] = mapped_column(Integer)

    __table_args__ = (
        Index(
            "ix_assets_code_issuer",
            "asset_code",
            func.coalesce(asset_issuer, ""),
            unique=True,
        ),
    )


# ---------------------------------------------------------------------------
# Graph mirror accounts
# ---------------------------------------------------------------------------

GRAPH_EDGE_TYPES = ("transaction", "claim", "payment")
GRAPH_ID_TYPE = BigInteger().with_variant(Integer(), "sqlite")


class GraphAccount(Base):
    """Canonical graph node table.

    This table is intentionally separate from ``accounts`` because the raw
    ``accounts`` table stores the latest Stellar account snapshot, while the
    graph mirror needs stable surrogate keys and observation timestamps for
    long-lived node/edge analytics.
    """

    __tablename__ = "graph_accounts"

    id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True, autoincrement=True)
    account_address: Mapped[str] = mapped_column(String(56), nullable=False, unique=True)
    account_type: Mapped[Optional[str]] = mapped_column(String(32))
    first_seen_at: Mapped[datetime] = mapped_column(nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    outgoing_edges: Mapped[list[GraphEdge]] = relationship(
        foreign_keys="GraphEdge.source_account_id",
        back_populates="source_account",
    )
    incoming_edges: Mapped[list[GraphEdge]] = relationship(
        foreign_keys="GraphEdge.destination_account_id",
        back_populates="destination_account",
    )

    __table_args__ = (
        Index("ix_graph_accounts_last_seen_at", "last_seen_at"),
        Index("ix_graph_accounts_account_type", "account_type"),
    )


# ---------------------------------------------------------------------------
# Graph mirror edges
# ---------------------------------------------------------------------------

class GraphEdge(Base):
    """Canonical directed edge table for the PostgreSQL graph mirror.

    Shared edge attributes live here so the table stays narrow and indexable for
    account timelines. Type-specific attributes move to dedicated detail tables
    to avoid a single sparse event table.
    """

    __tablename__ = "graph_edges"

    id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True, autoincrement=True)
    edge_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_account_id: Mapped[int] = mapped_column(
        GRAPH_ID_TYPE, ForeignKey("graph_accounts.id"), nullable=False
    )
    destination_account_id: Mapped[Optional[int]] = mapped_column(
        GRAPH_ID_TYPE, ForeignKey("graph_accounts.id")
    )
    asset_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("assets.id"))
    occurred_at: Mapped[datetime] = mapped_column(nullable=False)
    ledger_sequence: Mapped[Optional[int]] = mapped_column(Integer)
    event_index: Mapped[Optional[int]] = mapped_column(Integer)
    transaction_hash: Mapped[Optional[str]] = mapped_column(String(64))
    external_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    status: Mapped[Optional[str]] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    source_account: Mapped[GraphAccount] = relationship(
        foreign_keys=[source_account_id],
        back_populates="outgoing_edges",
    )
    destination_account: Mapped[Optional[GraphAccount]] = relationship(
        foreign_keys=[destination_account_id],
        back_populates="incoming_edges",
    )
    asset: Mapped[Optional[Asset]] = relationship()
    transaction_detail: Mapped[Optional[GraphTransactionDetail]] = relationship(
        back_populates="edge",
        cascade="all, delete-orphan",
        uselist=False,
    )
    claim_detail: Mapped[Optional[GraphClaimDetail]] = relationship(
        back_populates="edge",
        cascade="all, delete-orphan",
        uselist=False,
    )
    payment_detail: Mapped[Optional[GraphPaymentDetail]] = relationship(
        back_populates="edge",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (
        CheckConstraint(
            "edge_type IN ('transaction', 'claim', 'payment')",
            name="ck_graph_edges_edge_type",
        ),
        CheckConstraint(
            "source_account_id <> destination_account_id OR destination_account_id IS NULL",
            name="ck_graph_edges_distinct_accounts",
        ),
        UniqueConstraint(
            "edge_type",
            "external_event_id",
            name="uq_graph_edges_type_external_event_id",
        ),
        UniqueConstraint("id", "edge_type", name="uq_graph_edges_id_edge_type"),
        Index("ix_graph_edges_occurred_at", "occurred_at"),
        Index(
            "ix_graph_edges_source_occurred_at",
            "source_account_id",
            "occurred_at",
        ),
        Index(
            "ix_graph_edges_destination_occurred_at",
            "destination_account_id",
            "occurred_at",
        ),
        Index("ix_graph_edges_type_occurred_at", "edge_type", "occurred_at"),
        Index("ix_graph_edges_asset_occurred_at", "asset_id", "occurred_at"),
        Index(
            "ix_graph_edges_status_occurred_at",
            "status",
            "occurred_at",
        ),
        Index(
            "ix_graph_edges_tx_hash",
            "transaction_hash",
            postgresql_where=(transaction_hash.isnot(None)),
        ),
        Index(
            "ix_graph_edges_ledger_event",
            "ledger_sequence",
            "event_index",
        ),
    )


class GraphTransactionDetail(Base):
    """Subtype table for transaction-specific edge attributes."""

    __tablename__ = "graph_transaction_details"

    edge_id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True)
    edge_type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="transaction"
    )
    successful: Mapped[Optional[bool]] = mapped_column(Boolean)
    operation_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    fee: Mapped[Optional[int]] = mapped_column(BigInteger)
    memo_type: Mapped[Optional[str]] = mapped_column(String(16))
    memo: Mapped[Optional[str]] = mapped_column(Text)
    details: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )

    edge: Mapped[GraphEdge] = relationship(back_populates="transaction_detail")

    __table_args__ = (
        CheckConstraint(
            "edge_type = 'transaction'",
            name="ck_graph_transaction_details_edge_type",
        ),
        ForeignKeyConstraint(
            ["edge_id", "edge_type"],
            ["graph_edges.id", "graph_edges.edge_type"],
            ondelete="CASCADE",
        ),
    )


class GraphClaimDetail(Base):
    """Subtype table for claim-specific edge attributes."""

    __tablename__ = "graph_claim_details"

    edge_id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True)
    edge_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="claim")
    claim_reference: Mapped[Optional[str]] = mapped_column(String(128))
    claim_status: Mapped[Optional[str]] = mapped_column(String(32))
    expires_at: Mapped[Optional[datetime]] = mapped_column()
    details: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )

    edge: Mapped[GraphEdge] = relationship(back_populates="claim_detail")

    __table_args__ = (
        CheckConstraint(
            "edge_type = 'claim'",
            name="ck_graph_claim_details_edge_type",
        ),
        ForeignKeyConstraint(
            ["edge_id", "edge_type"],
            ["graph_edges.id", "graph_edges.edge_type"],
            ondelete="CASCADE",
        ),
        Index("ix_graph_claim_details_claim_status", "claim_status"),
    )


class GraphPaymentDetail(Base):
    """Subtype table for payment-specific edge attributes."""

    __tablename__ = "graph_payment_details"

    edge_id: Mapped[int] = mapped_column(GRAPH_ID_TYPE, primary_key=True)
    edge_type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="payment"
    )
    payment_reference: Mapped[Optional[str]] = mapped_column(String(128))
    payment_status: Mapped[Optional[str]] = mapped_column(String(32))
    fee_amount: Mapped[Optional[float]] = mapped_column(Numeric)
    settled_at: Mapped[Optional[datetime]] = mapped_column()
    details: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )

    edge: Mapped[GraphEdge] = relationship(back_populates="payment_detail")

    __table_args__ = (
        CheckConstraint(
            "edge_type = 'payment'",
            name="ck_graph_payment_details_edge_type",
        ),
        CheckConstraint(
            "fee_amount >= 0 OR fee_amount IS NULL",
            name="ck_graph_payment_details_fee_amount_non_negative",
        ),
        ForeignKeyConstraint(
            ["edge_id", "edge_type"],
            ["graph_edges.id", "graph_edges.edge_type"],
            ondelete="CASCADE",
        ),
        Index("ix_graph_payment_details_payment_status", "payment_status"),
    )


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

class Effect(Base):
    """One row per Stellar effect — captures state changes from operations.

    Effects represent the outcomes of operations on accounts, such as balance
    changes, signers, flags, and other state modifications. This table provides
    a more granular view of account state changes than operations alone.
    """

    __tablename__ = "effects"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account: Mapped[str] = mapped_column(String(56), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    asset_code: Mapped[Optional[str]] = mapped_column(String(12))
    asset_issuer: Mapped[Optional[str]] = mapped_column(String(56))
    destination_account: Mapped[Optional[str]] = mapped_column(String(56))
    created_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    details: Mapped[Optional[dict]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql")
    )

    __table_args__ = (
        Index("ix_effects_account_created_at", "account", "created_at"),
        Index("ix_effects_type_created_at", "type", "created_at"),
        Index("ix_effects_destination_created_at", "destination_account", "created_at"),
    )


# ---------------------------------------------------------------------------
# Normalized Transactions
# ---------------------------------------------------------------------------

class NormalizedTransaction(Base):
    """Normalized representation of a transaction operation."""

    __tablename__ = "normalized_transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transaction_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    sender: Mapped[str] = mapped_column(String(56), nullable=False)
    receiver: Mapped[Optional[str]] = mapped_column(String(56))
    asset: Mapped[str] = mapped_column(String(70), nullable=False)
    amount: Mapped[Optional[float]] = mapped_column(Numeric)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)

    __table_args__ = (
        Index("ix_normalized_transactions_hash", "transaction_hash"),
        Index("ix_normalized_transactions_sender_timestamp", "sender", "timestamp"),
        Index(
            "ix_normalized_transactions_receiver_timestamp",
            "receiver",
            "timestamp",
            postgresql_where=(receiver.isnot(None)),
        ),
    )
