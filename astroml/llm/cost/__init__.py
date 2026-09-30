"""LLM Cost and Budget Management Package."""

from __future__ import annotations

from astroml.llm.cost.analytics import forecast_cost, get_cost_summary
from astroml.llm.cost.budget import (
    BudgetExceededError,
    ModelAccessDeniedError,
    check_budget,
    set_emergency_override,
)
from astroml.llm.cost.optimizer import route_request
from astroml.llm.cost.tracker import calculate_cost, track_request

__all__ = [
    "track_request",
    "calculate_cost",
    "check_budget",
    "BudgetExceededError",
    "ModelAccessDeniedError",
    "set_emergency_override",
    "route_request",
    "get_cost_summary",
    "forecast_cost",
]
