"""LLM Cost Tracking Logic."""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from astroml.db.models.cost import LLMCostRecord, LLMBudget
from astroml.llm.cost.alerts import check_and_trigger_alerts

logger = logging.getLogger(__name__)

# Token cost rates per 1,000 tokens
MODEL_RATES = {
    # OpenAI
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    # Fallback/Default local model rates
    "local": {"input": 0.0001, "output": 0.0001},
    "huggingface": {"input": 0.0002, "output": 0.0002},
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate request cost based on model name and tokens."""
    model_key = model_name.lower()
    rates = MODEL_RATES.get(model_key)
    
    if not rates:
        # Try substring matching
        for k, v in MODEL_RATES.items():
            if k in model_key:
                rates = v
                break
        else:
            rates = MODEL_RATES["local"]
            
    input_cost = (input_tokens / 1000.0) * rates["input"]
    output_cost = (output_tokens / 1000.0) * rates["output"]
    return input_cost + output_cost


async def track_request(
    db: AsyncSession,
    user_id: str,
    feature: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    team_id: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> float:
    """Record LLM call usage, compute cost, and accumulate in user/team budget in real-time."""
    cost = calculate_cost(model_name, input_tokens, output_tokens)
    
    # 1. Create LLM Cost Record
    record = LLMCostRecord(
        user_id=user_id,
        team_id=team_id,
        feature=feature,
        model_name=model_name,
        prompt_template=prompt_template,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        latency_ms=latency_ms,
        timestamp=datetime.utcnow()
    )
    db.add(record)
    
    # 2. Accumulate to user budget
    await _accumulate_budget(db, user_id, "user", cost)
    
    # 3. Accumulate to team budget if applicable
    if team_id:
        await _accumulate_budget(db, team_id, "team", cost)
        
    await db.commit()
    return cost


async def _accumulate_budget(db: AsyncSession, entity_id: str, scope: str, cost: float) -> None:
    """Add cost to budget and check alerts."""
    result = await db.execute(
        select(LLMBudget).where(LLMBudget.entity_id == entity_id)
    )
    budget = result.scalar_one_or_none()
    
    if not budget:
        # Create default free tier budget if none exists
        budget = LLMBudget(
            entity_id=entity_id,
            scope=scope,
            tier="free",
            limit_amount=10.0,
            current_spend=0.0,
            period="monthly",
            is_blocked=False
        )
        db.add(budget)
        await db.flush()
        
    budget.current_spend += cost
    
    # Check if budget is exceeded and enforce hard stop if no override
    if budget.current_spend >= budget.limit_amount:
        if not budget.emergency_override:
            budget.is_blocked = True
            logger.warning(
                "LLM Budget Blocked: entity %s reached limit of $%.2f (spend: $%.2f)",
                entity_id, budget.limit_amount, budget.current_spend
            )
            
    # Check and trigger alerts (50%, 80%, 100%)
    await check_and_trigger_alerts(db, budget)
