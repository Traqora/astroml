"""Loyalty Points API — Issue #255.

Endpoints:
  GET  /api/v1/loyalty/{account_id}/summary   — tier, balance, next-tier info
  GET  /api/v1/loyalty/{account_id}/history   — paginated earning history
  POST /api/v1/loyalty/{account_id}/redeem    — redeem points atomically
  GET  /api/v1/loyalty/tiers                  — all tiers with thresholds
  GET  /api/v1/loyalty/{account_id}/referral  — referral link + stats
"""
from __future__ import annotations

import hashlib
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import cast, Date, func, select
from sqlalchemy.orm import Session

from api.schemas import (
    BenefitOut,
    LoyaltySummaryFull,
    LoyaltyTierOut,
    NextTierInfo,
    PointsHistoryResponse,
    PointsTransactionOut,
    RedeemRequest,
    RedeemResponse,
    ReferralOut,
)
from api.graphql import publish_loyalty_points

router = APIRouter(prefix="/api/v1/loyalty", tags=["loyalty"])

# ─── Static tier definitions ─────────────────────────────────────────────────

_TIERS = [
    LoyaltyTierOut(id="bronze",   name="Bronze",   threshold=0,    multiplier=1.0,  color="#cd7f32"),
    LoyaltyTierOut(id="silver",   name="Silver",   threshold=1500, multiplier=1.1,  color="#c0c0c0"),
    LoyaltyTierOut(id="gold",     name="Gold",     threshold=3000, multiplier=1.25, color="#d4af37"),
    LoyaltyTierOut(id="platinum", name="Platinum", threshold=6000, multiplier=1.5,  color="#e5e4e2"),
]

_BENEFITS = {
    "bronze":   [BenefitOut(id="b1", title="Basic Access", description="Access to standard features.")],
    "silver":   [BenefitOut(id="b1", title="Free Shipping", description="No shipping fees."),
                 BenefitOut(id="b2", title="Birthday Bonus", description="500 bonus points on birthday.")],
    "gold":     [BenefitOut(id="b1", title="Free Shipping", description="No shipping fees."),
                 BenefitOut(id="b2", title="Birthday Bonus", description="500 bonus points on birthday."),
                 BenefitOut(id="b3", title="Priority Support", description="Skip the queue.")],
    "platinum": [BenefitOut(id="b1", title="Free Shipping", description="No shipping fees."),
                 BenefitOut(id="b2", title="Birthday Bonus", description="1000 bonus points on birthday."),
                 BenefitOut(id="b3", title="Priority Support", description="Skip the queue."),
                 BenefitOut(id="b4", title="Dedicated Manager", description="Personal account manager.")],
}


def _tier_for(balance: int) -> LoyaltyTierOut:
    current = _TIERS[0]
    for tier in _TIERS:
        if balance >= tier.threshold:
            current = tier
    return current


def _next_tier(balance: int) -> Optional[NextTierInfo]:
    for tier in _TIERS:
        if balance < tier.threshold:
            prev_threshold = _tier_for(balance).threshold
            span = tier.threshold - prev_threshold
            progress = max(0, balance - prev_threshold)
            return NextTierInfo(
                tier=tier,
                remaining_to_upgrade=tier.threshold - balance,
                progress_pct=min(100, round(progress * 100 / span) if span else 100),
            )
    return None


# ─── DB dependency + ORM models ───────────────────────────────────────────────

def _get_db():
    try:
        from astroml.db.session import SessionLocal  # noqa: PLC0415
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    except ImportError:
        yield None


def _get_loyalty_models():
    """Lazy-import loyalty ORM models. Returns (LoyaltyAccount, PointsLedger) or (None, None)."""
    try:
        from api.loyalty_models import LoyaltyAccount, PointsLedger  # noqa: PLC0415
        return LoyaltyAccount, PointsLedger
    except ImportError:
        return None, None


def _get_or_create_account(account_id: str, db: Session):
    LoyaltyAccount, _ = _get_loyalty_models()
    if LoyaltyAccount is None:
        return None
    acc = db.get(LoyaltyAccount, account_id)
    if acc is None:
        acc = LoyaltyAccount(account_id=account_id, points_balance=0)
        db.add(acc)
        db.flush()
    return acc


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/tiers", response_model=list[LoyaltyTierOut])
def list_tiers():
    """List all loyalty tiers with thresholds and multipliers."""
    return _TIERS


@router.get("/{account_id}/summary", response_model=LoyaltySummaryFull)
def get_loyalty_summary(account_id: str, db: Optional[Session] = Depends(_get_db)):
    """Return current tier, points balance, and next-tier progress."""
    LoyaltyAccount, _ = _get_loyalty_models()
    balance = 0

    if db is not None and LoyaltyAccount is not None:
        acc = _get_or_create_account(account_id, db)
        db.commit()
        if acc:
            balance = acc.points_balance

    current = _tier_for(balance)
    return LoyaltySummaryFull(
        current_tier=current,
        points_balance=balance,
        next_tier=_next_tier(balance),
        benefits=_BENEFITS.get(current.id, []),
    )


@router.get("/{account_id}/history", response_model=PointsHistoryResponse)
def get_points_history(
    account_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Optional[Session] = Depends(_get_db),
):
    """Return paginated points earning/redemption history, sorted newest first."""
    _, PointsLedger = _get_loyalty_models()
    if db is None or PointsLedger is None:
        return PointsHistoryResponse(data=[], page=page, page_size=page_size, total=0)

    q = (
        select(PointsLedger)
        .where(PointsLedger.account_id == account_id)
        .order_by(PointsLedger.created_at.desc())
    )
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.offset((page - 1) * page_size).limit(page_size)).all()

    return PointsHistoryResponse(
        data=[
            PointsTransactionOut(
                id=str(r.id),
                date=r.created_at.isoformat(),
                type=r.txn_type,
                points=r.points,
                source=r.source,
                note=r.note,
            )
            for r in rows
        ],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.post("/{account_id}/redeem", response_model=RedeemResponse)
def redeem_points(
    account_id: str,
    body: RedeemRequest,
    db: Optional[Session] = Depends(_get_db),
):
    """Redeem points atomically. Validates balance, one-per-day limit, and minimum."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    LoyaltyAccount, PointsLedger = _get_loyalty_models()
    if LoyaltyAccount is None:
        raise HTTPException(status_code=503, detail="Loyalty service unavailable")

    with db.begin_nested() if db.in_transaction() else _noop_ctx():
        acc = _get_or_create_account(account_id, db)
        if acc is None:
            raise HTTPException(status_code=404, detail="Account not found")

        if body.points > acc.points_balance:
            raise HTTPException(status_code=400, detail="Insufficient points balance")

        if body.points < 100:
            raise HTTPException(status_code=400, detail="Minimum redemption is 100 points")

        # One redemption per day
        today_count = db.scalar(
            select(func.count(PointsLedger.id)).where(
                PointsLedger.account_id == account_id,
                PointsLedger.txn_type == "redeem",
                cast(PointsLedger.created_at, Date) == date.today(),
            )
        ) or 0
        if today_count >= 1:
            raise HTTPException(status_code=400, detail="One redemption allowed per day")

        acc.points_balance -= body.points
        # Recalculate tier (stored for denormalized reads)
        acc.tier_id = _tier_for(acc.points_balance).id

        txn_id = str(uuid.uuid4())
        ledger_row = PointsLedger(
            id=txn_id,
            account_id=account_id,
            txn_type="redeem",
            points=-body.points,
            source=f"reward:{body.reward_id}" if body.reward_id else "redemption",
            created_at=datetime.now(timezone.utc),
        )
        db.add(ledger_row)
        db.commit()

    return RedeemResponse(
        new_balance=acc.points_balance,
        transaction=PointsTransactionOut(
            id=txn_id,
            date=ledger_row.created_at.isoformat(),
            type="redeem",
            points=-body.points,
            source=ledger_row.source,
        ),
    )


@router.get("/{account_id}/referral", response_model=ReferralOut)
def get_referral(account_id: str, db: Optional[Session] = Depends(_get_db)):
    """Return referral link and stats for an account."""
    # Derive a deterministic referral code from account_id (no extra table needed)
    code = hashlib.sha256(account_id.encode()).hexdigest()[:8].upper()
    base_url = "https://astroml.example.com/ref"

    invited = 0
    rewards = 0
    if db is not None:
        _, PointsLedger = _get_loyalty_models()
        if PointsLedger is not None:
            rewards = db.scalar(
                select(func.count(PointsLedger.id)).where(
                    PointsLedger.account_id == account_id,
                    PointsLedger.source == f"referral:{code}",
                )
            ) or 0

    return ReferralOut(url=f"{base_url}?code={code}", invited=invited, rewards=rewards)


# ─── Helper ───────────────────────────────────────────────────────────────────

@contextmanager
def _noop_ctx():
    yield


# ─── Loyalty Points Update ──────────────────────────────────────────────────

async def update_loyalty_points(
    account_id: str,
    points_delta: int,
    source: str,
    note: Optional[str] = None,
    db: Optional[Session] = None,
) -> dict:
    """
    Update loyalty points for an account and publish to GraphQL subscriptions.

    Args:
        account_id: The account ID to update
        points_delta: Positive for earning, negative for redemption
        source: Source of the points change (e.g., "transaction", "referral")
        note: Optional note for the transaction
        db: Database session (creates one if not provided)

    Returns:
        Dict with updated balance, tier, and transaction ID

    Raises:
        HTTPException: If account not found or insufficient balance
    """
    own_db = False
    if db is None:
        db = next(_get_db())
        own_db = True

    try:
        LoyaltyAccount, PointsLedger = _get_loyalty_models()
        if LoyaltyAccount is None:
            raise HTTPException(status_code=503, detail="Loyalty service unavailable")

        acc = _get_or_create_account(account_id, db)
        if acc is None:
            raise HTTPException(status_code=404, detail="Account not found")

        # Check for sufficient balance if redeeming
        if points_delta < 0 and abs(points_delta) > acc.points_balance:
            raise HTTPException(status_code=400, detail="Insufficient points balance")

        # Update balance
        acc.points_balance += points_delta
        if acc.points_balance < 0:
            acc.points_balance = 0

        # Recalculate tier
        acc.tier_id = _tier_for(acc.points_balance).id

        # Create transaction record
        txn_type = "earn" if points_delta > 0 else "redeem"
        txn_id = str(uuid.uuid4())
        ledger_row = PointsLedger(
            id=txn_id,
            account_id=account_id,
            txn_type=txn_type,
            points=points_delta,
            source=source,
            note=note,
            created_at=datetime.now(timezone.utc),
        )
        db.add(ledger_row)
        db.commit()
        db.refresh(acc)

        # Publish to GraphQL subscription
        await publish_loyalty_points({
            "id": acc.id,
            "account_id": acc.account_id,
            "balance": acc.points_balance,
            "tier": acc.tier_id,
            "multiplier": _tier_for(acc.points_balance).multiplier,
            "updated_at": acc.updated_at.isoformat() if hasattr(acc, 'updated_at') else datetime.now(timezone.utc).isoformat(),
        })

        return {
            "success": True,
            "account_id": account_id,
            "new_balance": acc.points_balance,
            "tier": acc.tier_id,
            "transaction_id": txn_id,
        }

    finally:
        if own_db:
            db.close()


# ─── Exports ─────────────────────────────────────────────────────────────────

__all__ = [
    "router",
    "update_loyalty_points",
    "publish_loyalty_points",
]