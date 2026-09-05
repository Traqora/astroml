"""Experiments API router for AstroML.

Provides REST endpoints for experiment management, run comparison,
hyperparameter analysis, and experiment reporting.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.tracking.experiment_dashboard import ExperimentDashboard
from astroml.tracking.run_comparator import RunComparator, RunMetrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["experiments"])

_dashboard = ExperimentDashboard()
_comparator = RunComparator()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateExperimentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="")

    model_config = ConfigDict(extra="forbid")


class UpdateExperimentRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    notes: str | None = None
    tags: list[str] | None = None

    model_config = ConfigDict(extra="forbid")


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    description: str
    tags: list[str]
    notes: str
    created_at: str
    updated_at: str
    run_count: int

    model_config = ConfigDict(extra="forbid")


class RunMetricsRequest(BaseModel):
    run_name: str = Field(..., min_length=1)
    experiment_name: str = Field(default="default")
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    metric_history: dict[str, list[float]] = Field(default_factory=dict)
    status: str = Field(default="completed")

    model_config = ConfigDict(extra="forbid")


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(..., min_length=2)
    target_metric: str = Field(default="accuracy")
    higher_is_better: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class ReportResponse(BaseModel):
    experiment_id: str
    experiment_name: str
    generated_at: str
    num_runs: int
    best_run: dict[str, Any] | None = None
    metric_summary: dict[str, Any] = Field(default_factory=dict)
    param_importance: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = ""

    model_config = ConfigDict(extra="forbid")


class DashboardStatsResponse(BaseModel):
    total_experiments: int
    total_runs: int
    tag_counts: dict[str, int] = Field(default_factory=dict)
    recent_experiments: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class MessageResponse(BaseModel):
    message: str
    success: bool = True

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Experiment CRUD endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/experiments",
    response_model=ExperimentResponse,
    status_code=201,
    summary="Create a new experiment",
)
async def create_experiment(request: CreateExperimentRequest) -> ExperimentResponse:
    try:
        exp = _dashboard.create_experiment(
            name=request.name,
            description=request.description,
            tags=request.tags,
            notes=request.notes,
        )
        return _exp_to_response(exp)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("Error creating experiment")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/experiments",
    response_model=list[ExperimentResponse],
    summary="List experiments",
)
async def list_experiments(
    tag: list[str] | None = Query(default=None, description="Filter by tags"),
    search: str | None = Query(default=None, description="Search query"),
    sort_by: str = Query(default="created_at", pattern=r"^(created_at|updated_at|name)$"),
    reverse: bool = Query(default=False),
) -> list[ExperimentResponse]:
    try:
        exps = _dashboard.list_experiments(
            tag_filter=tag,
            search_query=search,
            sort_by=sort_by,
            reverse=reverse,
        )
        return [_exp_to_response(e) for e in exps]
    except Exception as e:
        logger.exception("Error listing experiments")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment by ID",
)
async def get_experiment(experiment_id: str) -> ExperimentResponse:
    exp = _dashboard.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _exp_to_response(exp)


@router.patch(
    "/experiments/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Update experiment",
)
async def update_experiment(
    experiment_id: str,
    request: UpdateExperimentRequest,
) -> ExperimentResponse:
    try:
        exp = _dashboard.update_experiment(
            experiment_id=experiment_id,
            name=request.name,
            description=request.description,
            notes=request.notes,
            tags=request.tags,
        )
        return _exp_to_response(exp)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error updating experiment")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/experiments/{experiment_id}",
    response_model=MessageResponse,
    summary="Delete experiment",
)
async def delete_experiment(experiment_id: str) -> MessageResponse:
    try:
        deleted = _dashboard.delete_experiment(experiment_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return MessageResponse(message=f"Deleted experiment {experiment_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting experiment")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run management endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/experiments/{experiment_id}/runs",
    response_model=MessageResponse,
    status_code=201,
    summary="Add a run to an experiment",
)
async def add_run(experiment_id: str, request: RunMetricsRequest) -> MessageResponse:
    try:
        import uuid as _uuid

        run = RunMetrics(
            run_id=_uuid.uuid4().hex[:12],
            run_name=request.run_name,
            experiment_name=request.experiment_name,
            metrics=request.metrics,
            params=request.params,
            tags=request.tags,
            metric_history=request.metric_history,
            status=request.status,
        )
        _dashboard.add_run(experiment_id, run)
        return MessageResponse(
            message=f"Added run {run.run_id} to experiment {experiment_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error adding run")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/experiments/{experiment_id}/runs/{run_id}",
    response_model=MessageResponse,
    summary="Remove a run from an experiment",
)
async def remove_run(experiment_id: str, run_id: str) -> MessageResponse:
    try:
        _dashboard.remove_run(experiment_id, run_id)
        return MessageResponse(
            message=f"Removed run {run_id} from experiment {experiment_id}",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error removing run")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Comparison endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/experiments/compare",
    summary="Compare multiple runs",
)
async def compare_runs(request: CompareRequest) -> dict[str, Any]:
    try:
        result = _dashboard.compare_runs(
            run_ids=request.run_ids,
            target_metric=request.target_metric,
            higher_is_better=request.higher_is_better,
        )
        return {
            "best_run": result.best_run,
            "worst_run": result.worst_run,
            "metric_diffs": {
                k: [{"run_a": a, "run_b": b, "diff": round(d, 6)} for a, b, d in v]
                for k, v in result.metric_diffs.items()
            },
            "correlation_scores": result.correlation_scores,
            "param_importance": [
                {"parameter": p, "importance": round(s, 6)}
                for p, s in result.param_importance
            ],
            "summary": result.summary,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error comparing runs")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/experiments/hyperparameter-importance",
    summary="Get hyperparameter importance scores",
)
async def hyperparameter_importance(
    run_ids: list[str] = Query(..., description="Run IDs to analyze"),
    target_metric: str = Query(default="accuracy"),
    top_k: int = Query(default=10, ge=1, le=50),
) -> list[dict[str, Any]]:
    try:
        scores = _dashboard.hyperparameter_importance(
            run_ids=run_ids,
            target_metric=target_metric,
            top_k=top_k,
        )
        return [
            {"parameter": p, "importance": round(s, 6)} for p, s in scores
        ]
    except Exception as e:
        logger.exception("Error computing hyperparameter importance")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Tagging endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/experiments/{experiment_id}/tags",
    response_model=ExperimentResponse,
    summary="Add a tag to an experiment",
)
async def add_tag(experiment_id: str, tag: str = Query(..., min_length=1)) -> ExperimentResponse:
    try:
        exp = _dashboard.add_tag(experiment_id, tag)
        return _exp_to_response(exp)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error adding tag")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/experiments/{experiment_id}/tags",
    response_model=ExperimentResponse,
    summary="Remove a tag from an experiment",
)
async def remove_tag(experiment_id: str, tag: str = Query(..., min_length=1)) -> ExperimentResponse:
    try:
        exp = _dashboard.remove_tag(experiment_id, tag)
        return _exp_to_response(exp)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error removing tag")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Cloning endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/experiments/{experiment_id}/clone",
    response_model=ExperimentResponse,
    status_code=201,
    summary="Clone an experiment",
)
async def clone_experiment(
    experiment_id: str,
    new_name: str | None = Query(default=None),
    copy_runs: bool = Query(default=False),
) -> ExperimentResponse:
    try:
        exp = _dashboard.clone_experiment(
            experiment_id=experiment_id,
            new_name=new_name,
            copy_runs=copy_runs,
        )
        return _exp_to_response(exp)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error cloning experiment")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Reporting endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/experiments/{experiment_id}/report",
    response_model=ReportResponse,
    summary="Generate experiment report",
)
async def generate_report(
    experiment_id: str,
    target_metric: str = Query(default="accuracy"),
    format: str = Query(default="json", pattern=r"^(json|markdown)$"),
) -> dict[str, Any]:
    try:
        if format == "markdown":
            md = _dashboard.export_report_markdown(experiment_id, target_metric)
            return {"content": md, "format": "markdown"}

        data = _dashboard.export_report_json(experiment_id, target_metric)
        return data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error generating report")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------


@router.get(
    "/experiments/stats",
    response_model=DashboardStatsResponse,
    summary="Get experiment dashboard statistics",
)
async def get_stats() -> DashboardStatsResponse:
    try:
        stats = _dashboard.dashboard_stats()
        return DashboardStatsResponse(**stats)
    except Exception as e:
        logger.exception("Error getting dashboard stats")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exp_to_response(exp) -> ExperimentResponse:
    return ExperimentResponse(
        experiment_id=exp.experiment_id,
        name=exp.name,
        description=exp.description,
        tags=exp.tags,
        notes=exp.notes,
        created_at=exp.created_at,
        updated_at=exp.updated_at,
        run_count=len(exp.runs),
    )