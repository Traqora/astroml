"""Fraud Detection API endpoints for AstroML.

Provides real-time fraud scoring, paginated alert listing, and
aggregated statistics.
"""
from __future__ import annotations

import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
import os

from astroml.api.models import FraudAlert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Async DB session dependency (shared config pattern matching app.py)
# ---------------------------------------------------------------------------

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://astroml:astroml@localhost/astroml",
)
_engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
_session_factory: async_sessionmaker = async_sessionmaker(
    _engine, expire_on_commit=False
)

_SLOW_REQUEST_THRESHOLD_MS = 500


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with _session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    """Request body for the fraud scoring endpoint."""

    accounts: List[str] = Field(..., description="List of Stellar public keys to score")
    edges: List[dict] = Field(default_factory=list, description="Transaction edges for graph context")

    @field_validator("accounts")
    @classmethod
    def accounts_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("accounts must be a non-empty list")
        if len(v) > 50:
            raise ValueError("accounts list may contain at most 50 entries")
        return v


class ScoreResponse(BaseModel):
    """Response body for the fraud scoring endpoint."""

    scores: Dict[str, float]


class FraudAlertItem(BaseModel):
    """Single fraud alert representation."""

    id: int
    account_id: str
    score: float
    risk_level: str
    batch_run_at: str
    created_at: str
    notes: Optional[str]

    model_config = {"from_attributes": True}


class FraudAlertListResponse(BaseModel):
    """Paginated list of fraud alerts."""

    items: List[FraudAlertItem]
    total: int
    page: int
    page_size: int


class FraudStatsResponse(BaseModel):
    """Aggregated fraud alert statistics."""

    total_alerts: int
    high: int
    medium: int
    low: int
    risk_over_time: List


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/api/v1/fraud/score", tags=["fraud"])
async def score_accounts(body: ScoreRequest) -> JSONResponse:
    """Score a batch of accounts for fraud risk.

    Returns a map of account_id -> anomaly score.  If models are not yet
    trained, returns 503 with an actionable error message.  Requests taking
    longer than 500 ms are logged as warnings.
    """
    start = time.monotonic()

    try:
        from astroml.pipeline.scoring import InductiveAnomalyScorer  # noqa: F401
    except (ImportError, Exception) as exc:
        logger.warning("Scoring pipeline unavailable: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": "Models not yet trained",
                "detail": "Run the training pipeline first",
            },
        )

    # Attempt to load a scorer; surface missing/corrupted checkpoints as 503.
    try:
        from astroml.pipeline.inductive import InductiveGraphSAGE
        from astroml.models.deep_svdd import DeepSVDD

        # We only produce scores if a trained scorer can be instantiated.
        # Constructors may raise FileNotFoundError / RuntimeError for missing
        # checkpoints; we catch broadly and return 503.
        pipeline = InductiveGraphSAGE()
        svdd = DeepSVDD()
        scorer = InductiveAnomalyScorer(pipeline=pipeline, svdd=svdd)

        import time as _time
        scores = scorer.score_new_accounts(
            edges=body.edges,
            account_ids=body.accounts,
            ref_time=_time.time(),
        )
    except Exception as exc:
        logger.warning("Model checkpoint unavailable or corrupted: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": "Models not yet trained",
                "detail": "Run the training pipeline first",
            },
        )

    elapsed_ms = (time.monotonic() - start) * 1000
    if elapsed_ms > _SLOW_REQUEST_THRESHOLD_MS:
        logger.warning("Fraud scoring took %.1f ms (threshold %d ms)", elapsed_ms, _SLOW_REQUEST_THRESHOLD_MS)

    return JSONResponse(content={"scores": scores})


@router.get(
    "/api/v1/fraud/alerts",
    response_model=FraudAlertListResponse,
    tags=["fraud"],
)
async def list_fraud_alerts(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    risk_level: Optional[str] = Query(
        default=None,
        description="Filter by risk level: low, medium, or high",
    ),
    db: AsyncSession = Depends(get_db),
) -> FraudAlertListResponse:
    """Return a paginated list of fraud alerts with optional risk_level filter."""
    base_stmt = select(FraudAlert)

    if risk_level is not None:
        base_stmt = base_stmt.where(FraudAlert.risk_level == risk_level)

    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    paged_stmt = (
        base_stmt.order_by(FraudAlert.created_at.desc()).offset(offset).limit(page_size)
    )
    rows = (await db.execute(paged_stmt)).scalars().all()

    items = [
        FraudAlertItem(
            id=row.id,
            account_id=row.account_id,
            score=row.score,
            risk_level=row.risk_level,
            batch_run_at=row.batch_run_at.isoformat(),
            created_at=row.created_at.isoformat(),
            notes=row.notes,
        )
        for row in rows
    ]

    return FraudAlertListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/api/v1/fraud/stats", response_model=FraudStatsResponse, tags=["fraud"])
async def fraud_stats(db: AsyncSession = Depends(get_db)) -> FraudStatsResponse:
    """Return aggregated fraud alert statistics."""
    total: int = (await db.execute(select(func.count(FraudAlert.id)))).scalar_one()

    high_count: int = (
        await db.execute(
            select(func.count(FraudAlert.id)).where(FraudAlert.risk_level == "high")
        )
    ).scalar_one()

    medium_count: int = (
        await db.execute(
            select(func.count(FraudAlert.id)).where(FraudAlert.risk_level == "medium")
        )
    ).scalar_one()

    low_count: int = (
        await db.execute(
            select(func.count(FraudAlert.id)).where(FraudAlert.risk_level == "low")
        )
    ).scalar_one()

    return FraudStatsResponse(
        total_alerts=total,
        high=high_count,
        medium=medium_count,
        low=low_count,
        risk_over_time=[],
    )
