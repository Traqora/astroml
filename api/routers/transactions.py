"""Transaction History API (issue #253).

Endpoints
---------
GET /api/v1/transactions        — List transactions with rich filtering
GET /api/v1/transactions/stats  — Aggregated stats (volume, count by asset)
GET /api/v1/transactions/{hash} — Single transaction by hash
GET /api/v1/transactions/{hash}/explain — LLM explanation of transaction

Query params for list endpoint:
  source_account, destination_account, asset_code, start_date, end_date,
  min_amount, max_amount, operation_type, successful, page, page_size
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models.orm import ApiTransaction as Transaction
from api.graphql import publish_transaction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from astroml.llm.explainer import TransactionExplainer

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])
explainer = TransactionExplainer()


# ─── Schemas ─────────────────────────────────────────────────────────────────

class TransactionOut(BaseModel):
    hash: str
    ledgerSequence: int
    sourceAccount: str
    destinationAccount: Optional[str] = None
    amount: Optional[float] = None
    assetCode: Optional[str] = None
    assetIssuer: Optional[str] = None
    fee: int
    operationType: Optional[str] = None
    successful: bool
    memoType: Optional[str] = None
    createdAt: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}

    @classmethod
    def from_orm(cls, obj):
        """Convert from ORM model with snake_case to camelCase."""
        return cls(
            hash=obj.hash,
            ledgerSequence=obj.ledger_sequence,
            sourceAccount=obj.source_account,
            destinationAccount=obj.destination_account,
            amount=float(obj.amount) if obj.amount is not None else None,
            assetCode=obj.asset_code,
            assetIssuer=obj.asset_issuer,
            fee=obj.fee,
            operationType=obj.operation_type,
            successful=obj.successful,
            memoType=obj.memo_type,
            createdAt=obj.created_at,
        )


class TransactionHistoryResponse(BaseModel):
    data: list[TransactionOut]
    page: int
    pageSize: int
    total: int


class TransactionStats(BaseModel):
    total_count: int
    total_volume: float
    count_by_asset: dict[str, int]
    successful_count: int
    failed_count: int


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=TransactionStats)
async def transaction_stats(db: AsyncSession = Depends(get_db)):
    """Aggregated transaction statistics."""
    total_count = (await db.execute(select(func.count()).select_from(Transaction))).scalar_one()
    total_volume = (await db.execute(
        select(func.coalesce(func.sum(Transaction.amount), 0))
    )).scalar_one()
    successful_count = (await db.execute(
        select(func.count()).where(Transaction.successful.is_(True))
    )).scalar_one()

    rows = (await db.execute(
        select(Transaction.asset_code, func.count())
        .group_by(Transaction.asset_code)
    )).all()
    count_by_asset = {(r[0] or "native"): r[1] for r in rows}

    return TransactionStats(
        total_count=total_count,
        total_volume=float(total_volume),
        count_by_asset=count_by_asset,
        successful_count=successful_count,
        failed_count=total_count - successful_count,
    )


@router.get("/{hash}", response_model=TransactionOut)
async def get_transaction(hash: str, db: AsyncSession = Depends(get_db)):
    """Fetch a single transaction by hash."""
    result = await db.execute(select(Transaction).where(Transaction.hash == hash))
    tx = result.scalar_one_or_none()
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return TransactionOut.from_orm(tx)


@router.get("/{hash}/explain")
async def explain_transaction(hash: str, db: AsyncSession = Depends(get_db)):
    """Explain a single transaction by hash using LLM."""
    result = await db.execute(select(Transaction).where(Transaction.hash == hash))
    tx = result.scalar_one_or_none()
    if tx is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
        
    tx_data = {
        'id': tx.hash,
        'from_address': tx.source_account,
        'to_address': tx.destination_account,
        'amount': str(tx.amount) if tx.amount is not None else '0'
    }
    
    explanation = explainer.explain(tx_data)
    return {"hash": hash, "explanation": explanation}


@router.get("", response_model=TransactionHistoryResponse)
async def list_transactions(
    source_account: Optional[str] = Query(None),
    destination_account: Optional[str] = Query(None),
    asset_code: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    operation_type: Optional[str] = Query(None),
    successful: Optional[bool] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """List transactions with optional compound filtering."""
    q = select(Transaction)

    if source_account:
        q = q.where(Transaction.source_account == source_account)
    if destination_account:
        q = q.where(Transaction.destination_account == destination_account)
    if asset_code:
        q = q.where(Transaction.asset_code == asset_code)
    if start_date:
        q = q.where(Transaction.created_at >= start_date)
    if end_date:
        q = q.where(Transaction.created_at <= end_date)
    if min_amount is not None:
        q = q.where(Transaction.amount >= min_amount)
    if max_amount is not None:
        q = q.where(Transaction.amount <= max_amount)
    if operation_type:
        q = q.where(Transaction.operation_type == operation_type)
    if successful is not None:
        q = q.where(Transaction.successful.is_(successful))

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar_one()

    q = q.order_by(Transaction.created_at.desc())
    q = q.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(q)).scalars().all()

    return TransactionHistoryResponse(
        data=[TransactionOut.from_orm(row) for row in rows],
        page=page,
        pageSize=page_size,
        total=total,
    )


# ─── Transaction Creation ────────────────────────────────────────────────────

async def create_transaction(tx_data: dict, db: AsyncSession):
    """Create a new transaction and publish to GraphQL subscriptions."""
    # Convert from API format to ORM format
    transaction = Transaction(
        hash=tx_data.get("hash"),
        ledger_sequence=tx_data.get("ledger_sequence"),
        source_account=tx_data.get("source_account"),
        destination_account=tx_data.get("destination_account"),
        amount=tx_data.get("amount"),
        asset_code=tx_data.get("asset_code"),
        asset_issuer=tx_data.get("asset_issuer"),
        fee=tx_data.get("fee", 0),
        operation_type=tx_data.get("operation_type"),
        successful=tx_data.get("successful", True),
        memo_type=tx_data.get("memo_type"),
        created_at=tx_data.get("created_at", datetime.utcnow()),
    )
    
    db.add(transaction)
    await db.commit()
    await db.refresh(transaction)
    
    # Publish to GraphQL subscription
    await publish_transaction({
        "hash": transaction.hash,
        "ledger_sequence": transaction.ledger_sequence,
        "source_account": transaction.source_account,
        "destination_account": transaction.destination_account,
        "amount": transaction.amount,
        "asset_code": transaction.asset_code,
        "asset_issuer": transaction.asset_issuer,
        "fee": transaction.fee,
        "operation_type": transaction.operation_type,
        "successful": transaction.successful,
        "memo_type": transaction.memo_type,
        "created_at": transaction.created_at,
    })
    
    return transaction