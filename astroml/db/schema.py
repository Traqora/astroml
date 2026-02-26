"""SQLAlchemy ORM models for raw Stellar blockchain data.

Five tables model the core Stellar data needed for graph ML:

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
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    SmallInteger,
    String,
    Text,
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
