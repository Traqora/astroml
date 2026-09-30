from astroml.utils.exceptions import AstroMLError
"""FastAPI router for model debugging and error analysis endpoints."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from astroml.training.debugging.confusion_analysis import ConfusionAnalyzer
from astroml.training.debugging.error_analysis import ErrorAnalyzer
from astroml.training.debugging.failure_modes import FailureModeIdentifier
from astroml.training.debugging.slice_analysis import SliceAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["debugging"])


# ─── Request / Response models ──────────────────────────────────────────────


class ErrorAnalysisRequest(BaseModel):
    """Request payload for error analysis."""

    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_pred: list[int]


class ConfusionRequest(BaseModel):
    """Request payload for confusion matrix analysis."""

    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_pred: list[int]
    labels: list[int] | None = None
    norm: str = "true"


class SliceRequest(BaseModel):
    """Request payload for slice-based performance analysis."""

    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_pred: list[int]
    slice_labels: list[str]
    threshold: float = 0.6


class FailureModesRequest(BaseModel):
    """Request payload for failure mode identification."""

    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_pred: list[int]
    y_prob: list[float] | None = None
    slice_labels: list[str] | None = None


class DebuggingResponse(BaseModel):
    """Generic response envelope for debugging endpoints."""

    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]


# ─── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/debugging/error-analysis", response_model=DebuggingResponse)
async def error_analysis(request: ErrorAnalysisRequest) -> DebuggingResponse:
    """Run error analysis on a set of predictions."""
    try:
        analyzer = ErrorAnalyzer()
        result = analyzer.analyze(np.array(request.y_true), np.array(request.y_pred))
        data = {
            "total_samples": result.total_samples,
            "error_count": result.error_count,
            "error_rate": result.error_rate,
            "accuracy": result.accuracy,
            "error_indices": result.error_indices.tolist(),
            "class_metrics": {
                str(label): {
                    "total": metrics.total,
                    "correct": metrics.correct,
                    "incorrect": metrics.incorrect,
                    "error_rate": metrics.error_rate,
                }
                for label, metrics in result.class_metrics.items()
            },
            "error_distribution": {
                f"{true}->{pred}": count
                for (true, pred), count in result.error_distribution.items()
            },
        }
        return DebuggingResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/debugging/confusion-matrix", response_model=DebuggingResponse)
async def confusion_matrix(request: ConfusionRequest) -> DebuggingResponse:
    """Compute a confusion matrix with per-class metrics."""
    try:
        analyzer = ConfusionAnalyzer()
        result = analyzer.compute(
            np.array(request.y_true),
            np.array(request.y_pred),
            labels=request.labels,
            norm=request.norm,
        )
        return DebuggingResponse(status="success", data=analyzer.to_dict(result))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Confusion matrix computation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/debugging/slices", response_model=DebuggingResponse)
async def slice_analysis(request: SliceRequest) -> DebuggingResponse:
    """Analyze model performance across data slices."""
    try:
        analyzer = SliceAnalyzer()
        result = analyzer.analyze(
            np.array(request.y_true),
            np.array(request.y_pred),
            np.array(request.slice_labels),
        )
        underperforming = analyzer.underperforming_slices(
            np.array(request.y_true),
            np.array(request.y_pred),
            np.array(request.slice_labels),
            threshold=request.threshold,
        )
        data = {
            "overall_accuracy": result.overall_accuracy,
            "worst_slice": result.worst_slice,
            "worst_accuracy": result.worst_accuracy,
            "slices": [
                {
                    "slice_name": metrics.slice_name,
                    "support": metrics.support,
                    "accuracy": metrics.accuracy,
                    "error_rate": metrics.error_rate,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
                for metrics in result.slices
            ],
            "underperforming_slices": [metrics.slice_name for metrics in underperforming],
        }
        return DebuggingResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Slice analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/debugging/failure-modes", response_model=DebuggingResponse)
async def failure_modes(request: FailureModesRequest) -> DebuggingResponse:
    """Identify failure modes in a model's predictions."""
    try:
        identifier = FailureModeIdentifier()
        result = identifier.identify(
            np.array(request.y_true),
            np.array(request.y_pred),
            y_prob=np.array(request.y_prob) if request.y_prob is not None else None,
            slice_labels=(
                np.array(request.slice_labels) if request.slice_labels is not None else None
            ),
        )
        return DebuggingResponse(status="success", data=identifier.summarize(result))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failure mode identification failed")
        raise HTTPException(status_code=500, detail=str(exc))
