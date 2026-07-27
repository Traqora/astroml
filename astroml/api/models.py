"""ORM models specific to the fraud-detection API layer.

``FraudAlert`` is kept in this module (rather than ``astroml/db/schema.py``)
so that the API package can be imported independently of the full ingestion
schema.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# SQLite does not support BigInteger autoincrement via RETURNING; use Integer
# on SQLite and BigInteger on PostgreSQL/other dialects.
_ID_TYPE = BigInteger().with_variant(Integer(), "sqlite")


class APIBase(DeclarativeBase):
    """Declarative base for API-layer ORM models."""


class FraudAlert(APIBase):
    """One row per fraud-scoring result produced by the batch scheduler.

    Columns
    -------
    id              Auto-incremented surrogate key.
    account_id      Stellar account address (G…, 56 chars).
    score           Anomaly score — higher values are more suspicious.
    risk_level      Bucketed label: ``low``, ``medium``, or ``high``.
    batch_run_at    Timestamp of the batch run that produced this row.
    created_at      Row-insertion timestamp (server default).
    notes           Optional free-text notes (e.g. reason for flagging).
    """

    __tablename__ = "fraud_alerts"

    id: Mapped[int] = mapped_column(_ID_TYPE, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(String(56), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    batch_run_at: Mapped[datetime] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    notes: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        Index("ix_fraud_alerts_account_id", "account_id"),
        Index("ix_fraud_alerts_batch_run_at", "batch_run_at"),
        Index("ix_fraud_alerts_risk_level", "risk_level"),
        Index("ix_fraud_alerts_created_at", "created_at"),
    )

    @staticmethod
    def risk_level_for_score(score: float) -> str:
        """Bucket a numeric anomaly score into a labelled risk level."""
        if score >= 0.8:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"
