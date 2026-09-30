from astroml.utils.exceptions import AstroMLError
"""FastAPI router for automated model selection and architecture search."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from astroml.training.model_selection.automl import AutoMLConfig, AutoMLPipeline
from astroml.training.model_selection.meta_learning import MetaLearningRecommender
from astroml.training.model_selection.nas import NeuralArchitectureSearch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["model-selection"])


# ─── Request / Response models ──────────────────────────────────────────────


class DataRequest(BaseModel):
    """Request payload carrying a small labeled dataset."""

    model_config = ConfigDict(extra="forbid")

    X: list[list[float]]
    y: list[int]


class AutoMLRequest(DataRequest):
    """Request payload for AutoML search."""

    model_config = ConfigDict(extra="forbid")

    cv: int = 5
    scoring: str = "accuracy"
    max_models: int = 6
    time_budget: float = 0.0


class NASRequest(DataRequest):
    """Request payload for neural architecture search."""

    model_config = ConfigDict(extra="forbid")

    n_candidates: int = 10
    cv: int = 3
    time_budget: float = 0.0


class ModelSelectionResponse(BaseModel):
    """Generic response envelope for model selection endpoints."""

    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]


# ─── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/model-selection/automl", response_model=ModelSelectionResponse)
async def automl_search(request: AutoMLRequest) -> ModelSelectionResponse:
    """Run an AutoML search and return the ranked leaderboard."""
    try:
        pipeline = AutoMLPipeline()
        config = AutoMLConfig(
            cv=request.cv,
            scoring=request.scoring,
            max_models=request.max_models,
            time_budget=request.time_budget,
        )
        result = pipeline.search(np.array(request.X), np.array(request.y), config)
        data = {
            "best_model": result.best_model_name,
            "best_score": result.best_score,
            "searched": result.searched,
            "params": result.params,
            "leaderboard": [
                {
                    "model": entry.model_name,
                    "cv_mean": entry.cv_mean,
                    "cv_std": entry.cv_std,
                    "fit_time": entry.fit_time,
                    "predict_time": entry.predict_time,
                }
                for entry in result.leaderboard
            ],
        }
        return ModelSelectionResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("AutoML search failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/model-selection/nas", response_model=ModelSelectionResponse)
async def nas_search(request: NASRequest) -> ModelSelectionResponse:
    """Search for the best neural network architecture."""
    try:
        search = NeuralArchitectureSearch()
        result = search.search(
            np.array(request.X),
            np.array(request.y),
            n_candidates=request.n_candidates,
            cv=request.cv,
            time_budget=request.time_budget,
        )
        data = {
            "best_architecture": result.best_architecture.to_dict(),
            "best_score": result.best_score,
            "evaluated": result.evaluated,
            "candidates": [
                {"architecture": spec.to_dict(), "score": score}
                for spec, score in result.candidates
            ],
        }
        return ModelSelectionResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("NAS search failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/model-selection/recommend", response_model=ModelSelectionResponse)
async def recommend_model(request: DataRequest) -> ModelSelectionResponse:
    """Recommend a model using meta-learning heuristics."""
    try:
        recommender = MetaLearningRecommender()
        model_name, score, confidence = recommender.recommend(
            np.array(request.X), np.array(request.y)
        )
        descriptor = recommender.describe(np.array(request.X), np.array(request.y))
        data = {
            "recommended_model": model_name,
            "expected_score": score,
            "confidence": confidence,
            "task_description": {
                "n_samples": descriptor.n_samples,
                "n_features": descriptor.n_features,
                "n_classes": descriptor.n_classes,
                "imbalance_ratio": descriptor.imbalance_ratio,
            },
        }
        return ModelSelectionResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Model recommendation failed")
        raise HTTPException(status_code=500, detail=str(exc))
