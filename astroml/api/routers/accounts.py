"""Account API endpoints for AstroML.

Provides paginated account listing, single-account lookup, transaction
history, fraud summary, and loyalty information.
"""
from __future__ import annotations

from datetime import datetime
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
import os

from astroml.api.models import FraudAlert
from astroml.db.schema import Account, Transaction

# ---------------------------------------------------------------------------
# Async DB session dependency
# ---------------------------------------------------------------------------

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://astroml:astroml@localhost/astroml",
)
_engine = create_async_engine(_DATABASE_URL, pool_pre_ping=True)
_session_factory: async_sessionmaker = async_sessionmaker(
    _engine, expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with _session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class AccountItem(BaseModel):
    """Summary representation of a single account."""

    account_id: str
    balance_xlm: Optional[float]
    sequence_number: Optional[int]
    updated_at: Optional[datetime]

    model_config = {"from_attributes": True}


class AccountListResponse(BaseModel):
    """Paginated list of accounts."""

    items: List[AccountItem]
    total: int
    page: int
    page_size: int


class TransactionItem(BaseModel):
    """Summary representation of a single transaction."""

    hash: str
    ledger_sequence: int
    source_account: str
    created_at: datetime
    fee: int
    operation_count: int
    successful: bool
    memo_type: Optional[str]
    memo: Optional[str]

    model_config = {"from_attributes": True}


class TransactionListResponse(BaseModel):
    """Paginated list of transactions."""

    items: List[TransactionItem]
    total: int
    page: int
    page_size: int


class FraudSummaryResponse(BaseModel):
    """Fraud alert summary for an account."""

    account_id: str
    total_alerts: int
    high: int
    medium: int
    low: int
    latest_score: Optional[float]
    latest_risk_level: Optional[str]


class LoyaltyResponse(BaseModel):
    """Loyalty information for an account."""

    account_id: str
    points: int
    tier: str
    message: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/api/v1/accounts", response_model=AccountListResponse, tags=["accounts"])
async def list_accounts(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    public_key: Optional[str] = Query(default=None, description="Filter by account public key"),
    date_from: Optional[datetime] = Query(default=None, description="Filter accounts updated on or after this datetime (ISO 8601)"),
    date_to: Optional[datetime] = Query(default=None, description="Filter accounts updated on or before this datetime (ISO 8601)"),
    db: AsyncSession = Depends(get_db),
) -> AccountListResponse:
    """List accounts with optional filters and pagination."""
    stmt = select(Account)

    if public_key is not None:
        stmt = stmt.where(Account.account_id == public_key)
    if date_from is not None:
        stmt = stmt.where(Account.updated_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(Account.updated_at <= date_to)

    # Count total matching rows
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    # Fetch the requested page
    offset = (page - 1) * page_size
    paged_stmt = stmt.offset(offset).limit(page_size)
    rows = (await db.execute(paged_stmt)).scalars().all()

    items = [
        AccountItem(
            account_id=row.account_id,
            balance_xlm=float(row.balance) if row.balance is not None else None,
            sequence_number=row.sequence,
            updated_at=row.updated_at,
        )
        for row in rows
    ]

    return AccountListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/api/v1/accounts/{public_key}", response_model=AccountItem, tags=["accounts"])
async def get_account(
    public_key: str,
    db: AsyncSession = Depends(get_db),
) -> AccountItem:
    """Get a single account by public key."""
    stmt = select(Account).where(Account.account_id == public_key)
    row = (await db.execute(stmt)).scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Account {public_key!r} not found")

    return AccountItem(
        account_id=row.account_id,
        balance_xlm=float(row.balance) if row.balance is not None else None,
        sequence_number=row.sequence,
        updated_at=row.updated_at,
    )


@router.get(
    "/api/v1/accounts/{public_key}/transactions",
    response_model=TransactionListResponse,
    tags=["accounts"],
)
async def get_account_transactions(
    public_key: str,
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> TransactionListResponse:
    """Get paginated transactions for an account, ordered by created_at desc."""
    base_stmt = select(Transaction).where(Transaction.source_account == public_key)

    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * page_size
    paged_stmt = (
        base_stmt.order_by(Transaction.created_at.desc()).offset(offset).limit(page_size)
    )
    rows = (await db.execute(paged_stmt)).scalars().all()

    items = [TransactionItem.model_validate(row) for row in rows]
    return TransactionListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get(
    "/api/v1/accounts/{public_key}/fraud-summary",
    response_model=FraudSummaryResponse,
    tags=["accounts"],
)
async def get_account_fraud_summary(
    public_key: str,
    db: AsyncSession = Depends(get_db),
) -> FraudSummaryResponse:
    """Return a fraud alert summary for the given account."""
    base_stmt = select(FraudAlert).where(FraudAlert.account_id == public_key)

    total: int = (
        await db.execute(select(func.count()).select_from(base_stmt.subquery()))
    ).scalar_one()

    def _count_by_level(level: str) -> int:
        return 0  # filled below via individual queries

    high_count: int = (
        await db.execute(
            select(func.count()).select_from(
                select(FraudAlert)
                .where(FraudAlert.account_id == public_key, FraudAlert.risk_level == "high")
                .subquery()
            )
        )
    ).scalar_one()

    medium_count: int = (
        await db.execute(
            select(func.count()).select_from(
                select(FraudAlert)
                .where(FraudAlert.account_id == public_key, FraudAlert.risk_level == "medium")
                .subquery()
            )
        )
    ).scalar_one()

    low_count: int = (
        await db.execute(
            select(func.count()).select_from(
                select(FraudAlert)
                .where(FraudAlert.account_id == public_key, FraudAlert.risk_level == "low")
                .subquery()
            )
        )
    ).scalar_one()

    # Fetch the latest alert for score and risk_level
    latest_stmt = (
        select(FraudAlert)
        .where(FraudAlert.account_id == public_key)
        .order_by(FraudAlert.created_at.desc())
        .limit(1)
    )
    latest = (await db.execute(latest_stmt)).scalar_one_or_none()

    return FraudSummaryResponse(
        account_id=public_key,
        total_alerts=total,
        high=high_count,
        medium=medium_count,
        low=low_count,
        latest_score=latest.score if latest else None,
        latest_risk_level=latest.risk_level if latest else None,
    )


@router.get(
    "/api/v1/accounts/{public_key}/loyalty",
    response_model=LoyaltyResponse,
    tags=["accounts"],
)
async def get_account_loyalty(public_key: str) -> LoyaltyResponse:
    """Return loyalty info for an account (placeholder)."""
    return LoyaltyResponse(
        account_id=public_key,
        points=0,
        tier="bronze",
        message="Loyalty system coming soon",
    )
