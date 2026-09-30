from astroml.utils.exceptions import AstroMLError
"""FastAPI router for fairness evaluation and bias mitigation endpoints."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from astroml.validation.fairness.bias_detector import BiasDetector
from astroml.validation.fairness.metrics import FairnessMetrics
from astroml.validation.fairness.mitigation import BiasMitigation
from astroml.validation.fairness.report import FairnessReport

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["fairness"])


# ─── Request / Response models ──────────────────────────────────────────────


class MetricsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_pred: list[int]
    sensitive_features: list[int]


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]


class BiasDetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_pred: list[int]
    sensitive_features: list[list[int]] | list[int]
    attributes: list[str] | None = None

    @field_validator("sensitive_features", mode="before")
    @classmethod
    def coerce_sensitive_features(cls, v: Any) -> list[list[int]]:
        if v and isinstance(v, list) and len(v) > 0 and isinstance(v[0], int):
            return [[x] for x in v]
        return v


class BiasDetectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]


class IntersectionalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_pred: list[int]
    sensitive_features: list[list[int]]
    intersection_groups: list[list[int]] | None = None


class IntersectionalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    data: list[dict[str, Any]]


class MitigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    X: list[list[float]]
    y: list[int]
    sensitive_features: list[int]
    strategy: str = "reweighing"
    y_pred: list[float] | None = None


class MitigateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_pred: list[int]
    sensitive_features: list[list[int]] | list[int]
    attributes: list[str] | None = None

    @field_validator("sensitive_features", mode="before")
    @classmethod
    def coerce_sensitive_features(cls, v: Any) -> list[list[int]]:
        if v and isinstance(v, list) and len(v) > 0 and isinstance(v[0], int):
            return [[x] for x in v]
        return v


class ReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/fairness/metrics", response_model=MetricsResponse)
async def compute_metrics(request: MetricsRequest) -> MetricsResponse:
    """Compute fairness metrics for binary classification."""
    try:
        metrics = FairnessMetrics()
        y_true = np.array(request.y_true)
        y_pred = np.array(request.y_pred)
        sensitive = np.array(request.sensitive_features)

        results = metrics.compute_all(y_true, y_pred, sensitive)
        data: dict[str, Any] = {}
        for name, result in results.items():
            data[name] = {
                "value": result.value,
                "passed": result.passed,
                "threshold": result.threshold,
                "group_metrics": result.group_metrics,
            }
        return MetricsResponse(status="success", data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AstroMLError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fairness/bias/detect", response_model=BiasDetectResponse)
async def detect_bias(request: BiasDetectRequest) -> BiasDetectResponse:
    """Detect bias across protected attributes."""
    try:
        detector = BiasDetector()
        y_true = np.array(request.y_true)
        y_pred = np.array(request.y_pred)
        sensitive = np.array(request.sensitive_features)

        result = detector.detect_bias(y_true, y_pred, sensitive, attributes=request.attributes)
        report_data = detector.report_bias(result)
        return BiasDetectResponse(status="success", data=report_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AstroMLError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fairness/bias/intersectional", response_model=IntersectionalResponse)
async def intersectional_analysis(
    request: IntersectionalRequest,
) -> IntersectionalResponse:
    """Perform intersectional bias analysis."""
    try:
        detector = BiasDetector()
        y_true = np.array(request.y_true)
        y_pred = np.array(request.y_pred)
        sensitive = np.array(request.sensitive_features)

        results = detector.intersectional_analysis(
            y_true, y_pred, sensitive, intersection_groups=request.intersection_groups
        )
        data = [
            {
                "groups": list(r.groups),
                "bias_detected": r.bias_detected,
                "sample_size": r.sample_size,
            }
            for r in results
        ]
        return IntersectionalResponse(status="success", data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AstroMLError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fairness/mitigate", response_model=MitigateResponse)
async def mitigate_bias(request: MitigateRequest) -> MitigateResponse:
    """Apply bias mitigation strategy."""
    try:
        mitigator = BiasMitigation()
        X = np.array(request.X)
        y = np.array(request.y)
        sensitive = np.array(request.sensitive_features)
        y_pred = np.array(request.y_pred) if request.y_pred is not None else None

        kwargs: dict[str, Any] = {}
        if y_pred is not None:
            kwargs["base_predictions"] = y_pred

        result = mitigator.mitigate(X, y, sensitive, strategy=request.strategy, **kwargs)
        data = {
            "strategy_used": result.strategy_used,
            "before_metrics": result.before_metrics,
            "after_metrics": result.after_metrics,
            "improvement": result.improvement,
        }
        return MitigateResponse(status="success", data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AstroMLError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fairness/report", response_model=ReportResponse)
async def generate_fairness_report(
    request: ReportRequest,
) -> ReportResponse:
    """Generate comprehensive fairness report."""
    try:
        metrics = FairnessMetrics()
        detector = BiasDetector()
        report = FairnessReport()

        y_true = np.array(request.y_true)
        y_pred = np.array(request.y_pred)
        sensitive = np.array(request.sensitive_features)

        bias_result = detector.detect_bias(y_true, y_pred, sensitive, attributes=request.attributes)

        if sensitive.ndim == 2 and sensitive.shape[1] > 1:
            fairness_metrics = metrics.compute_all(y_true, y_pred, sensitive[:, 0])
        else:
            fairness_metrics = metrics.compute_all(y_true, y_pred, sensitive.ravel())

        report.generate_report(fairness_metrics=fairness_metrics, bias_results=bias_result)
        return ReportResponse(status="success", data=report.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AstroMLError as e:
        raise HTTPException(status_code=500, detail=str(e))
