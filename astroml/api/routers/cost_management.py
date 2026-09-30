"""Cost management API: spend tracking, budgets and optimization advice.

Resolves part of #647.  Mounted at ``/api/v1/cost``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.tracking.budget_manager import (
    DEFAULT_THRESHOLDS,
    Budget,
    BudgetExceededError,
    BudgetManager,
    BudgetPeriod,
)
from astroml.tracking.cost_tracker import CostTracker, ResourceType, default_cost_tracker
from astroml.tracking.resource_optimizer import ResourceOptimizer, ResourceUsageSample

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cost", tags=["cost-management"])

_tracker: CostTracker = default_cost_tracker
_budgets = BudgetManager(_tracker)
_optimizer = ResourceOptimizer(_tracker)


def get_tracker() -> CostTracker:
    """Return the tracker backing this router (FastAPI dependency / test hook)."""
    return _tracker


def get_budget_manager() -> BudgetManager:
    """Return the budget manager backing this router."""
    return _budgets


def get_optimizer() -> ResourceOptimizer:
    """Return the resource optimizer backing this router."""
    return _optimizer


# ─── Schemas ─────────────────────────────────────────────────────────────────


class ComputeCostRequest(BaseModel):
    """Record consumption of a compute resource."""

    model_config = ConfigDict(extra="forbid")

    resource: ResourceType
    hours: float = Field(ge=0.0)
    count: float = Field(default=1.0, ge=0.0)
    project: str | None = None
    team: str | None = None
    model_name: str | None = None
    cost_usd: float | None = Field(default=None, ge=0.0)


class ApiCostRequest(BaseModel):
    """Record calls made against an external paid service."""

    model_config = ConfigDict(extra="forbid")

    service: str = Field(min_length=1)
    calls: int = Field(ge=0)
    project: str | None = None
    team: str | None = None
    model_name: str | None = None
    cost_usd: float | None = Field(default=None, ge=0.0)


class CostRecordResponse(BaseModel):
    """A recorded cost entry."""

    model_config = ConfigDict(extra="forbid")

    category: str
    resource: str
    quantity: float
    unit: str
    cost_usd: float
    project: str | None = None
    team: str | None = None
    model_name: str | None = None
    timestamp: datetime


class BudgetRequest(BaseModel):
    """Create or replace a budget."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    limit_usd: float = Field(gt=0.0)
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    project: str | None = None
    team: str | None = None
    thresholds: list[float] = Field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    hard_limit: bool = False


class SpendCheckRequest(BaseModel):
    """Ask whether a prospective spend fits within the configured budgets."""

    model_config = ConfigDict(extra="forbid")

    additional_usd: float = Field(ge=0.0)
    project: str | None = None
    team: str | None = None


class UtilizationSampleRequest(BaseModel):
    """Report a resource utilization observation."""

    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(min_length=1)
    resource_type: ResourceType
    utilization: float = Field(ge=0.0, le=1.0)
    memory_utilization: float | None = Field(default=None, ge=0.0, le=1.0)
    project: str | None = None
    team: str | None = None
    interruptible: bool = False


def _record_response(record: Any) -> CostRecordResponse:
    """Adapt a :class:`CostRecord` to its API representation."""
    return CostRecordResponse(
        category=record.category.value,
        resource=record.resource,
        quantity=record.quantity,
        unit=record.unit,
        cost_usd=record.cost_usd,
        project=record.project,
        team=record.team,
        model_name=record.model,
        timestamp=record.timestamp,
    )


# ─── Cost recording & reporting ──────────────────────────────────────────────


@router.post("/compute", response_model=CostRecordResponse, status_code=201)
async def record_compute_cost(request: ComputeCostRequest) -> CostRecordResponse:
    """Record compute spend (GPU/CPU hours)."""
    record = get_tracker().record_compute(
        request.resource,
        request.hours,
        count=request.count,
        project=request.project,
        team=request.team,
        model=request.model_name,
        cost_usd=request.cost_usd,
    )
    return _record_response(record)


@router.post("/api-usage", response_model=CostRecordResponse, status_code=201)
async def record_api_cost(request: ApiCostRequest) -> CostRecordResponse:
    """Record spend against an external API."""
    record = get_tracker().record_api(
        request.service,
        request.calls,
        project=request.project,
        team=request.team,
        model=request.model_name,
        cost_usd=request.cost_usd,
    )
    return _record_response(record)


@router.get("/summary")
async def cost_summary(
    project: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    """Return total spend with per-category, per-resource and daily breakdowns."""
    filters: dict[str, Any] = {}
    if project is not None:
        filters["project"] = project
    if team is not None:
        filters["team"] = team
    return get_tracker().summary(**filters)


@router.get("/allocation")
async def cost_allocation() -> dict[str, dict[str, float]]:
    """Return spend allocated by project and by team."""
    return get_tracker().allocation()


@router.get("/forecast")
async def cost_forecast(
    horizon_days: int = Query(default=30, ge=1, le=365),
    lookback_days: int = Query(default=7, ge=1, le=365),
) -> dict[str, Any]:
    """Project spend over ``horizon_days`` from recent history."""
    return get_tracker().forecast(horizon_days=horizon_days, lookback_days=lookback_days).to_dict()


@router.get("/breakdown/{dimension}")
async def cost_breakdown(
    dimension: Literal["category", "resource", "project", "team", "model", "day"],
) -> dict[str, float]:
    """Return spend grouped along a single dimension."""
    return get_tracker().breakdown(dimension)


# ─── Budgets ─────────────────────────────────────────────────────────────────


@router.post("/budgets", status_code=201)
async def create_budget(request: BudgetRequest) -> dict[str, Any]:
    """Create or replace a budget."""
    try:
        budget = Budget(
            name=request.name,
            limit_usd=request.limit_usd,
            period=request.period,
            project=request.project,
            team=request.team,
            thresholds=tuple(request.thresholds),
            hard_limit=request.hard_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return get_budget_manager().add_budget(budget).to_dict()


@router.get("/budgets")
async def list_budgets() -> dict[str, Any]:
    """Return the status of every configured budget."""
    return get_budget_manager().report()


@router.get("/budgets/{name}")
async def budget_status(name: str) -> dict[str, Any]:
    """Return the status of one budget."""
    try:
        return get_budget_manager().status(name).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown budget {name!r}") from exc


@router.delete("/budgets/{name}", status_code=204)
async def delete_budget(name: str) -> None:
    """Delete a budget."""
    if not get_budget_manager().remove_budget(name):
        raise HTTPException(status_code=404, detail=f"unknown budget {name!r}")


@router.post("/budgets/evaluate")
async def evaluate_budgets() -> dict[str, Any]:
    """Evaluate all budgets and return alerts raised by this call."""
    alerts = get_budget_manager().evaluate()
    return {"alert_count": len(alerts), "alerts": [a.to_dict() for a in alerts]}


@router.get("/alerts")
async def list_alerts() -> dict[str, Any]:
    """Return the retained budget alert history."""
    alerts = get_budget_manager().alerts()
    return {"alert_count": len(alerts), "alerts": [a.to_dict() for a in alerts]}


@router.post("/budgets/check")
async def check_spend(request: SpendCheckRequest) -> dict[str, Any]:
    """Check a prospective spend against hard-limit budgets.

    Returns ``402 Payment Required`` when a hard limit would be breached.
    """
    try:
        statuses = get_budget_manager().check_spend(
            request.additional_usd, project=request.project, team=request.team
        )
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    return {"allowed": True, "budgets": [s.to_dict() for s in statuses]}


# ─── Resource optimization ───────────────────────────────────────────────────


@router.post("/utilization", status_code=202)
async def record_utilization(request: UtilizationSampleRequest) -> dict[str, str]:
    """Record a resource utilization sample for the optimizer."""
    get_optimizer().record(
        ResourceUsageSample(
            resource_id=request.resource_id,
            resource_type=request.resource_type,
            utilization=request.utilization,
            memory_utilization=request.memory_utilization,
            project=request.project,
            team=request.team,
            interruptible=request.interruptible,
        )
    )
    return {"status": "accepted"}


@router.get("/recommendations")
async def optimization_recommendations() -> dict[str, Any]:
    """Return ranked cost-optimization recommendations."""
    return get_optimizer().report()
