"""ORM models specific to the fraud-detection API layer.

``FraudAlert`` is kept in this module (rather than ``astroml/db/schema.py``)
so that the API package can be imported independently of the full ingestion
schema.
"""

from __future__ import annotations

from datetime import datetime

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

# Default anomaly-score thresholds for the low/medium/high buckets — issue
# #838. These are deliberately asset-agnostic: a score of 0.8 means the same
# thing whether the underlying transaction was in XLM or a stablecoin.
#
# Per-asset overrides exist because a fixed threshold is wrong for assets
# whose normal activity looks nothing alike — a stablecoin used for routine,
# large, regular transfers has a different "suspicious" baseline than a
# thinly-traded custom asset, where the same score represents a much rarer
# pattern. Overriding here means a model trained on scores alone can still
# be bucketed sensibly per asset without retraining or rescaling the score
# itself.
DEFAULT_RISK_THRESHOLDS: dict[str, float] = {"high": 0.8, "medium": 0.5}

# Asset code -> its own {"high": ..., "medium": ...} override. An asset
# absent from this map uses ``DEFAULT_RISK_THRESHOLDS`` unchanged, so adding
# an asset here only ever narrows or widens that one asset's buckets — it
# never changes behavior for every other asset already in production.
ASSET_RISK_THRESHOLDS: dict[str, dict[str, float]] = {}


def risk_level_for_score(score: float, asset: str | None = None) -> str:
    """Bucket a numeric anomaly score into a labelled risk level.

    ``asset`` is optional and defaults to the global thresholds
    (``DEFAULT_RISK_THRESHOLDS``) when omitted or when the given asset has
    no entry in ``ASSET_RISK_THRESHOLDS`` — every existing caller that does
    not pass ``asset`` keeps scoring exactly as before this parameter
    existed.
    """
    if asset is not None:
        thresholds = ASSET_RISK_THRESHOLDS.get(asset, DEFAULT_RISK_THRESHOLDS)
    else:
        thresholds = DEFAULT_RISK_THRESHOLDS
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["medium"]:
        return "medium"
    return "low"


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
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    notes: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_fraud_alerts_account_id", "account_id"),
        Index("ix_fraud_alerts_batch_run_at", "batch_run_at"),
        Index("ix_fraud_alerts_risk_level", "risk_level"),
        Index("ix_fraud_alerts_created_at", "created_at"),
    )

    @staticmethod
    def risk_level_for_score(score: float, asset: str | None = None) -> str:
        """Bucket a numeric anomaly score into a labelled risk level.

        ``asset`` is optional (issue #838) — see
        :func:`risk_level_for_score` (module level) for the per-asset
        threshold lookup this delegates to.
        """
        return risk_level_for_score(score, asset)
