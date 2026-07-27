"""Cost API Endpoints."""
from __future__ import annotations

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.database import get_db
from api.auth.dependencies import AuthContext, get_current_auth, require_scopes
from astroml.llm.cost import (
    get_cost_summary,
    forecast_cost,
    check_budget,
    set_emergency_override,
)
from astroml.db.models.cost import LLMBudget

router = APIRouter(prefix="/api/v1/cost", tags=["cost"])


@router.get("/summary")
async def get_summary_endpoint(
    days: int = 30,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve cost summary for the authenticated user."""
    user_id = str(auth.user_id or auth.subject)
    summary = await get_cost_summary(db, user_id, days)
    return summary


@router.get("/forecast")
async def get_forecast_endpoint(
    days: int = 30,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get spend forecast for the next N days."""
    user_id = str(auth.user_id or auth.subject)
    forecast = await forecast_cost(db, user_id, days)
    return forecast


@router.post("/budget")
async def configure_budget_endpoint(
    limit_amount: float,
    tier: str = "free",
    period: str = "monthly",
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Configure or update budget limit and tier for the authenticated user."""
    user_id = str(auth.user_id or auth.subject)
    
    # Fetch or create budget
    result = await db.execute(
        select(LLMBudget).where(LLMBudget.entity_id == user_id)
    )
    budget = result.scalar_one_or_none()
    
    if not budget:
        budget = LLMBudget(
            entity_id=user_id,
            scope="user",
            tier=tier,
            limit_amount=limit_amount,
            current_spend=0.0,
            period=period,
            is_blocked=False
        )
        db.add(budget)
    else:
        budget.limit_amount = limit_amount
        budget.tier = tier
        budget.period = period
        if budget.current_spend < limit_amount:
            budget.is_blocked = False
            
    await db.commit()
    return {
        "status": "success",
        "entity_id": user_id,
        "limit_amount": budget.limit_amount,
        "tier": budget.tier,
        "is_blocked": budget.is_blocked,
    }


@router.post("/override")
async def admin_override_endpoint(
    entity_id: str,
    override: bool = True,
    auth: AuthContext = Depends(require_scopes("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Enable/disable emergency override for a budget (Admin only)."""
    success = await set_emergency_override(db, entity_id, override)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget for entity '{entity_id}' not found."
        )
    return {"status": "success", "entity_id": entity_id, "emergency_override": override}
