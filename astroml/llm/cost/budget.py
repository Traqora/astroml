"""Budget enforcement rules and checks."""
from __future__ import annotations

import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from astroml.db.models.cost import LLMBudget

logger = logging.getLogger(__name__)

# Allowed models per tier
TIER_ALLOWED_MODELS = {
    "free": {"gpt-3.5-turbo", "local", "huggingface"},
    "pro": {"gpt-3.5-turbo", "gpt-4", "gpt-4o", "claude-3-sonnet", "claude-3-haiku", "local", "huggingface"},
    "enterprise": {"*"},  # all models allowed
}


class BudgetExceededError(Exception):
    """Exception raised when LLM budget limit has been reached."""
    pass


class ModelAccessDeniedError(Exception):
    """Exception raised when user tier does not permit accessing a model."""
    pass


async def is_model_allowed(tier: str, model_name: str) -> bool:
    """Verify if the requested model is allowed for the given budget tier."""
    allowed = TIER_ALLOWED_MODELS.get(tier.lower(), TIER_ALLOWED_MODELS["free"])
    if "*" in allowed:
        return True
        
    model_lower = model_name.lower()
    for m in allowed:
        if m in model_lower:
            return True
    return False


async def check_budget(
    db: AsyncSession,
    user_id: str,
    model_name: str,
    team_id: Optional[str] = None,
) -> bool:
    """
    Validate if a request should be allowed.
    Raises BudgetExceededError or ModelAccessDeniedError if blocked.
    """
    # 1. Check User Budget
    user_result = await db.execute(
        select(LLMBudget).where(LLMBudget.entity_id == user_id)
    )
    user_budget = user_result.scalar_one_or_none()
    
    # Setup default free budget if none exists
    if not user_budget:
        user_budget = LLMBudget(
            entity_id=user_id,
            scope="user",
            tier="free",
            limit_amount=10.0,
            current_spend=0.0,
            is_blocked=False,
            emergency_override=False
        )
        db.add(user_budget)
        await db.flush()
        
    # Check model compatibility with tier (unless admin override)
    if not user_budget.emergency_override:
        allowed = await is_model_allowed(user_budget.tier, model_name)
        if not allowed:
            raise ModelAccessDeniedError(
                f"Model '{model_name}' is not allowed on user tier '{user_budget.tier}'. "
                f"Upgrade to a higher tier to access this model."
            )
            
        if user_budget.is_blocked or user_budget.current_spend >= user_budget.limit_amount:
            raise BudgetExceededError(
                f"User LLM budget limit of ${user_budget.limit_amount:.2f} reached. "
                f"Current spend: ${user_budget.current_spend:.2f}."
            )
            
    # 2. Check Team Budget
    if team_id:
        team_result = await db.execute(
            select(LLMBudget).where(LLMBudget.entity_id == team_id)
        )
        team_budget = team_result.scalar_one_or_none()
        if team_budget and not team_budget.emergency_override:
            if team_budget.is_blocked or team_budget.current_spend >= team_budget.limit_amount:
                raise BudgetExceededError(
                    f"Team LLM budget limit of ${team_budget.limit_amount:.2f} reached. "
                    f"Current spend: ${team_budget.current_spend:.2f}."
                )
                
    return True


async def set_emergency_override(
    db: AsyncSession,
    entity_id: str,
    override: bool = True,
) -> bool:
    """Set emergency override for admin bypass."""
    result = await db.execute(
        select(LLMBudget).where(LLMBudget.entity_id == entity_id)
    )
    budget = result.scalar_one_or_none()
    if budget:
        budget.emergency_override = override
        if override:
            budget.is_blocked = False  # unblock if override is enabled
        await db.commit()
        return True
    return False
