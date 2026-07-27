"""ORM models for the Loyalty Points system."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_ID_TYPE = BigInteger().with_variant(Integer(), "sqlite")


class LoyaltyBase(DeclarativeBase):
    pass


class LoyaltyAccount(LoyaltyBase):
    """Loyalty account state: current balance and tier."""

    __tablename__ = "loyalty_accounts"

    account_id: Mapped[str] = mapped_column(String(56), primary_key=True)
    points_balance: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    tier_id: Mapped[str] = mapped_column(String(16), nullable=False, server_default="bronze")
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_loyalty_accounts_tier_id", "tier_id"),
    )


class PointsLedger(LoyaltyBase):
    """Immutable ledger of every points earn/redeem/adjust event."""

    __tablename__ = "points_ledger"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    account_id: Mapped[str] = mapped_column(String(56), nullable=False)
    txn_type: Mapped[str] = mapped_column(String(16), nullable=False)  # earn|redeem|adjust
    points: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(128))
    note: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_points_ledger_account_created_at", "account_id", "created_at"),
        Index("ix_points_ledger_txn_type", "txn_type"),
    )
