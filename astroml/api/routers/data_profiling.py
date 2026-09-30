from astroml.utils.exceptions import AstroMLError
"""FastAPI router for automated data profiling and exploratory analysis."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from astroml.preprocessing.profiling.data_profiler import DataProfiler
from astroml.preprocessing.profiling.insights import InsightGenerator
from astroml.preprocessing.profiling.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["data-profiling"])


# ─── Request / Response models ──────────────────────────────────────────────


class ProfilingRequest(BaseModel):
    """Request payload for data profiling.

    Provide either ``data`` (a list of row dictionaries) or ``columns``
    (a mapping of column name to list of values).
    """

    model_config = ConfigDict(extra="forbid")

    data: list[dict[str, Any]] | None = None
    columns: dict[str, list[Any]] | None = None


class ReportRequest(ProfilingRequest):
    """Request payload for report generation."""

    model_config = ConfigDict(extra="forbid")

    format: str = "html"


class ProfilingResponse(BaseModel):
    """Generic response envelope for profiling endpoints."""

    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]


def _build_dataframe(request: ProfilingRequest) -> pd.DataFrame:
    """Build a DataFrame from a profiling request payload.

    Args:
        request: The request containing either row dicts or column lists.

    Returns:
        The constructed DataFrame.

    Raises:
        HTTPException: If no data is supplied.
    """
    if request.data is not None:
        return pd.DataFrame(request.data)
    if request.columns is not None:
        return pd.DataFrame(request.columns)
    raise HTTPException(status_code=400, detail="Either 'data' or 'columns' must be provided")


def _profile(request: ProfilingRequest) -> dict[str, Any]:
    """Profile a request's data and return serializable results.

    Args:
        request: The profiling request.

    Returns:
        A dictionary with profiles, insights and quality score.
    """
    df = _build_dataframe(request)
    profiler = DataProfiler()
    result = profiler.profile(df)
    insights = InsightGenerator().generate(result)
    return {
        "row_count": result.row_count,
        "column_count": result.column_count,
        "duplicate_rows": result.duplicate_rows,
        "missing_total": result.missing_total,
        "quality_score": result.quality_score,
        "columns": {name: column.to_dict() for name, column in result.columns.items()},
        "insights": [insight.to_dict() for insight in insights],
        "insight_summary": InsightGenerator().summarize(insights),
    }


# ─── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/profiling/analyze", response_model=ProfilingResponse)
async def analyze_data(request: ProfilingRequest) -> ProfilingResponse:
    """Profile a dataset and return statistics, insights and a quality score."""
    try:
        return ProfilingResponse(status="success", data=_profile(request))
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Data profiling failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/profiling/report", response_model=ProfilingResponse)
async def generate_report(request: ReportRequest) -> ProfilingResponse:
    """Generate a profiling report (html, markdown, json, pdf)."""
    try:
        df = _build_dataframe(request)
        profiler = DataProfiler()
        result = profiler.profile(df)
        insights = InsightGenerator().generate(result)
        profile_data = {
            "row_count": result.row_count,
            "column_count": result.column_count,
            "duplicate_rows": result.duplicate_rows,
            "missing_total": result.missing_total,
            "quality_score": result.quality_score,
            "columns": {name: column.to_dict() for name, column in result.columns.items()},
            "insights": [insight.to_dict() for insight in insights],
            "insight_summary": InsightGenerator().summarize(insights),
        }
        report = ReportGenerator().generate(result, insights, fmt=request.format)
        return ProfilingResponse(
            status="success",
            data={"format": request.format, "report": report, **profile_data},
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
