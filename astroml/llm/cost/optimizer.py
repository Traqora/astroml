"""Model routing and cost optimization logic."""
from __future__ import annotations

import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from astroml.db.models.cost import LLMBudget

logger = logging.getLogger(__name__)


def select_optimal_model(
    current_spend: float,
    limit_amount: float,
    preferred_model: str,
    prompt_length: int,
    complexity: str = "medium",  # "low", "medium", "high"
) -> str:
    """
    Route to cheaper models dynamically when appropriate to save costs.
    If spend is close to the budget limits (>75%), or if complexity/prompt size is small,
    down-route to a cheaper model.
    """
    model_lower = preferred_model.lower()
    
    # Calculate budget utilization
    utilization = current_spend / limit_amount if limit_amount > 0 else 0.0
    
    # 1. Budget Squeeze Routing: if we are at >80% budget utilization, use cheaper models
    if utilization >= 0.8:
        if "gpt-4" in model_lower:
            logger.info("Optimizing cost: Routing from GPT-4 to GPT-3.5-turbo due to high budget utilization (%.1f%%)", utilization * 100)
            return "gpt-3.5-turbo"
        if "opus" in model_lower:
            logger.info("Optimizing cost: Routing from Claude-3-Opus to Claude-3-Haiku due to high budget utilization (%.1f%%)", utilization * 100)
            return "claude-3-haiku"
            
    # 2. Complexity-based Routing: if task has low complexity, route to a cheaper model
    if complexity == "low":
        if "gpt-4" in model_lower:
            logger.info("Optimizing cost: Routing from GPT-4 to gpt-3.5-turbo for low-complexity task")
            return "gpt-3.5-turbo"
        if "opus" in model_lower:
            logger.info("Optimizing cost: Routing from Claude-3-Opus to Claude-3-Haiku for low-complexity task")
            return "claude-3-haiku"
            
    # 3. Prompt length-based routing: if prompt is extremely short/simple
    if prompt_length < 100 and complexity != "high":
        if "gpt-4" in model_lower:
            return "gpt-3.5-turbo"
            
    return preferred_model


async def route_request(
    db: AsyncSession,
    user_id: str,
    preferred_model: str,
    prompt_text: str,
    complexity: str = "medium",
) -> str:
    """Route LLM request to optimal model based on budget spend and query complexity."""
    result = await db.execute(
        select(LLMBudget).where(LLMBudget.entity_id == user_id)
    )
    budget = result.scalar_one_or_none()
    
    current_spend = 0.0
    limit_amount = 10.0
    
    if budget:
        current_spend = budget.current_spend
        limit_amount = budget.limit_amount
        
    prompt_len = len(prompt_text)
    
    # Route optimal model
    optimal_model = select_optimal_model(
        current_spend=current_spend,
        limit_amount=limit_amount,
        preferred_model=preferred_model,
        prompt_length=prompt_len,
        complexity=complexity,
    )
    
    return optimal_model
