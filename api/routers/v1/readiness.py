"""Supported v1 inference readiness endpoint (issue #1224).

The endpoint reports whether the deployed fraud scorer can serve predictions.
It intentionally avoids the database, GraphQL and LLM router graphs so it can
be monitored as a small, stable v1 surface.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException

from api.schemas import ModelReadinessResponse
from api.services.scorer import load_scorer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ops"])
DEFAULT_CHECKPOINT = "benchmark_results/gcn_model.pt"


def _get_scorer() -> Any:
    return load_scorer()


def _checkpoint_path() -> str:
    return os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT)


@router.get("/fraud/ready", response_model=ModelReadinessResponse)
def fraud_ready() -> ModelReadinessResponse:
    """Return 200 when the fraud scorer is loaded, otherwise 503."""
    checkpoint = _checkpoint_path()
    try:
        scorer = _get_scorer()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model readiness check failed for %s: %s", checkpoint, exc)
        scorer = None

    if scorer is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "fail",
                "service": "fraud",
                "reason": "model_unavailable",
                "checkpoint": checkpoint,
                "remediation": "Set MODEL_CHECKPOINT_PATH to a valid checkpoint or activate a model.",
            },
        )

    return ModelReadinessResponse(checkpoint=checkpoint)
