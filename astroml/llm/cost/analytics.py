"""LLM Cost Analytics and Reporting."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from astroml.db.models.cost import LLMCostRecord


async def get_cost_summary(
    db: AsyncSession,
    user_id: str,
    days: int = 30,
) -> Dict[str, Any]:
    """Get aggregated cost and token usage summary for the last N days."""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Base query filter
    base_filter = and_(
        LLMCostRecord.user_id == user_id,
        LLMCostRecord.timestamp >= start_date
    )
    
    # 1. Total summary
    summary_query = select(
        func.sum(LLMCostRecord.cost).label("total_cost"),
        func.sum(LLMCostRecord.input_tokens).label("total_input_tokens"),
        func.sum(LLMCostRecord.output_tokens).label("total_output_tokens"),
        func.avg(LLMCostRecord.latency_ms).label("avg_latency_ms"),
        func.count(LLMCostRecord.id).label("request_count")
    ).where(base_filter)
    
    res = await db.execute(summary_query)
    row = res.fetchone()
    
    total_cost = float(row.total_cost or 0.0)
    input_tokens = int(row.total_input_tokens or 0)
    output_tokens = int(row.total_output_tokens or 0)
    avg_latency = float(row.avg_latency_ms or 0.0)
    request_count = int(row.request_count or 0)
    
    # 2. Per-feature breakdown
    feature_query = select(
        LLMCostRecord.feature,
        func.sum(LLMCostRecord.cost).label("cost"),
        func.count(LLMCostRecord.id).label("requests")
    ).where(base_filter).group_by(LLMCostRecord.feature)
    
    feature_res = await db.execute(feature_query)
    features = [
        {"feature": r.feature, "cost": float(r.cost or 0.0), "requests": r.requests}
        for r in feature_res.fetchall()
    ]
    
    # 3. Per-model breakdown
    model_query = select(
        LLMCostRecord.model_name,
        func.sum(LLMCostRecord.cost).label("cost"),
        func.sum(LLMCostRecord.input_tokens + LLMCostRecord.output_tokens).label("tokens")
    ).where(base_filter).group_by(LLMCostRecord.model_name)
    
    model_res = await db.execute(model_query)
    models = [
        {"model": r.model_name, "cost": float(r.cost or 0.0), "tokens": int(r.tokens or 0)}
        for r in model_res.fetchall()
    ]
    
    # 4. Per-template breakdown
    template_query = select(
        LLMCostRecord.prompt_template,
        func.sum(LLMCostRecord.cost).label("cost"),
        func.count(LLMCostRecord.id).label("requests")
    ).where(base_filter).group_by(LLMCostRecord.prompt_template)
    
    temp_res = await db.execute(template_query)
    templates = [
        {"template": r.prompt_template or "direct", "cost": float(r.cost or 0.0), "requests": r.requests}
        for r in temp_res.fetchall()
    ]
    
    # 5. Historical daily costs
    daily_query = select(
        func.date(LLMCostRecord.timestamp).label("day"),
        func.sum(LLMCostRecord.cost).label("cost")
    ).where(base_filter).group_by(func.date(LLMCostRecord.timestamp)).order_by("day")
    
    daily_res = await db.execute(daily_query)
    history = [
        {"date": str(r.day), "cost": float(r.cost or 0.0)}
        for r in daily_res.fetchall()
    ]
    
    return {
        "total_cost": total_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "avg_latency_ms": avg_latency,
        "request_count": request_count,
        "features": features,
        "models": models,
        "templates": templates,
        "history": history,
    }


async def forecast_cost(
    db: AsyncSession,
    user_id: str,
    days_to_forecast: int = 30,
) -> Dict[str, Any]:
    """Forecast future cost based on last 7 days of spending history."""
    summary = await get_cost_summary(db, user_id, days=7)
    history = summary["history"]
    
    if not history:
        return {"forecasted_spend": 0.0, "confidence": "low", "daily_average": 0.0}
        
    total_spend_7d = sum(h["cost"] for h in history)
    daily_avg = total_spend_7d / len(history)
    
    forecasted = daily_avg * days_to_forecast
    
    return {
        "forecasted_spend": round(forecasted, 4),
        "daily_average": round(daily_avg, 4),
        "confidence": "medium" if len(history) >= 5 else "low",
        "basis_days": len(history),
    }
