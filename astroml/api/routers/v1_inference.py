"""Small, versioned inference API for local baseline predictors."""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.util import find_spec
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

MAX_PREDICTION_ROWS = 256
MAX_FEATURES_PER_ROW = 256
MAX_PREDICTION_VALUES = 65_536

router = APIRouter(prefix="/api/v1", tags=["v1-inference"])

Predictor = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _mean_baseline(features: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.mean(features, axis=1, dtype=np.float64)


_PREDICTORS: dict[str, Predictor] = {"mean_baseline": _mean_baseline}


def register_predictor(name: str, predictor: Predictor) -> None:
    """Register a local scoring callable for ``POST /api/v1/predict``."""
    if not name or len(name) > 128:
        raise ValueError("predictor name must contain 1 to 128 characters")
    if not callable(predictor):
        raise TypeError("predictor must be callable")
    _PREDICTORS[name] = predictor


def unregister_predictor(name: str) -> bool:
    """Remove a predictor and return whether it was registered."""
    return _PREDICTORS.pop(name, None) is not None


class PredictionRequest(BaseModel):
    """A bounded batch of same-width feature rows to score."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="mean_baseline", min_length=1, max_length=128)
    features: list[
        Annotated[list[float], Field(min_length=1, max_length=MAX_FEATURES_PER_ROW)]
    ] = Field(min_length=1, max_length=MAX_PREDICTION_ROWS)

    @model_validator(mode="after")
    def validate_matrix(self) -> PredictionRequest:
        width = len(self.features[0])
        if any(len(row) != width for row in self.features):
            raise ValueError("features must be a rectangular matrix")
        if len(self.features) * width > MAX_PREDICTION_VALUES:
            raise ValueError(
                f"features may contain at most {MAX_PREDICTION_VALUES} values"
            )
        if not np.isfinite(np.asarray(self.features, dtype=np.float64)).all():
            raise ValueError("features must contain only finite numbers")
        return self


class PredictionResponse(BaseModel):
    """Stable v1 scalar-score response; scores are not calibrated probabilities."""

    schema_version: Literal["v1"] = "v1"
    model_name: str
    count: int
    scores: list[float]


class ReadinessResponse(BaseModel):
    """Local runtime and predictor readiness; external services are not probed."""

    status: Literal["ready", "not_ready"]
    required_dependencies: dict[str, bool]
    models: list[str]


@router.get("/health", response_model=dict[str, str])
async def health() -> dict[str, str]:
    """Liveness probe; does not require a registered predictor or external service."""
    return {"status": "ok"}


@router.get("/ready", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse | JSONResponse:
    """Readiness probe for the local numerical runtime and predictor registry."""
    models = sorted(_PREDICTORS)
    required_dependencies = {
        "numpy": find_spec("numpy") is not None,
        "registered_predictor": bool(models),
    }
    ready = all(required_dependencies.values())
    response = ReadinessResponse(
        status="ready" if ready else "not_ready",
        required_dependencies=required_dependencies,
        models=models,
    )
    if not ready:
        return JSONResponse(status_code=503, content=response.model_dump())
    return response


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Return one finite scalar score per feature row."""
    predictor = _PREDICTORS.get(request.model_name)
    if predictor is None:
        raise HTTPException(status_code=404, detail="requested model is not registered")

    features = np.asarray(request.features, dtype=np.float64)
    try:
        scores = np.asarray(predictor(features), dtype=np.float64)
    except Exception as exc:
        logger.exception("Registered predictor %r failed", request.model_name)
        raise HTTPException(status_code=500, detail="predictor execution failed") from exc

    if scores.ndim != 1 or scores.shape[0] != features.shape[0]:
        raise HTTPException(
            status_code=500,
            detail="predictor must return exactly one score per input row",
        )
    if not np.isfinite(scores).all():
        raise HTTPException(status_code=500, detail="predictor returned a non-finite score")

    return PredictionResponse(
        model_name=request.model_name,
        count=int(scores.shape[0]),
        scores=scores.tolist(),
    )