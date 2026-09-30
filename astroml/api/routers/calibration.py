from astroml.utils.exceptions import AstroMLError
"""FastAPI router for model calibration and uncertainty estimation endpoints."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import brier_score_loss

from astroml.training.calibration.bayesian import BayesianUncertainty
from astroml.training.calibration.conformal import ConformalPredictor
from astroml.training.calibration.isotonic import IsotonicCalibrator
from astroml.training.calibration.platt import PlattCalibrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["calibration"])


# ─── Request / Response models ──────────────────────────────────────────────


class CalibrateRequest(BaseModel):
    """Request payload for probability calibration."""

    model_config = ConfigDict(extra="forbid")

    y_true: list[int]
    y_prob: list[float]


class ConformalRequest(BaseModel):
    """Request payload for conformal prediction."""

    model_config = ConfigDict(extra="forbid")

    probs_cal: list[list[float]]
    y_cal: list[int]
    probs: list[list[float]]
    y_true: list[int] | None = None
    significance: float = 0.1


class UncertaintyRequest(BaseModel):
    """Request payload for uncertainty estimation."""

    model_config = ConfigDict(extra="forbid")

    probs: list[list[float]]


class CalibrationResponse(BaseModel):
    """Generic response envelope for calibration endpoints."""

    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]


# ─── Endpoints ──────────────────────────────────────────────────────────────


@router.post("/calibration/platt", response_model=CalibrationResponse)
async def platt_calibration(request: CalibrateRequest) -> CalibrationResponse:
    """Fit Platt scaling and return calibrated probabilities."""
    try:
        calibrator = PlattCalibrator()
        y_prob = np.array(request.y_prob)
        y_true = np.array(request.y_true)
        calibrator.fit(y_prob, y_true)
        calibrated = calibrator.calibrate(y_prob)
        data = {
            "a": calibrator.a,
            "b": calibrator.b,
            "calibrated_probabilities": calibrated.tolist(),
            "brier_before": brier_score_loss(y_true, y_prob),
            "brier_after": brier_score_loss(y_true, calibrated),
        }
        return CalibrationResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Platt calibration failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/calibration/isotonic", response_model=CalibrationResponse)
async def isotonic_calibration(request: CalibrateRequest) -> CalibrationResponse:
    """Fit isotonic regression and return calibrated probabilities."""
    try:
        calibrator = IsotonicCalibrator()
        y_prob = np.array(request.y_prob)
        y_true = np.array(request.y_true)
        calibrator.fit(y_prob, y_true)
        calibrated = calibrator.calibrate(y_prob)
        data = {
            "calibrated_probabilities": calibrated.tolist(),
            "brier_before": brier_score_loss(y_true, y_prob),
            "brier_after": brier_score_loss(y_true, calibrated),
        }
        return CalibrationResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Isotonic calibration failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/calibration/conformal", response_model=CalibrationResponse)
async def conformal_prediction(request: ConformalRequest) -> CalibrationResponse:
    """Build conformal prediction sets with empirical coverage."""
    try:
        predictor = ConformalPredictor()
        predictor.fit_classification(np.array(request.probs_cal), np.array(request.y_cal))
        result = predictor.predict_sets(
            np.array(request.probs),
            significance=request.significance,
            y_true=np.array(request.y_true) if request.y_true is not None else None,
        )
        data = {
            "prediction_sets": result.prediction_sets,
            "quantile": result.quantile,
            "coverage": result.coverage,
            "classes": result.classes,
        }
        return CalibrationResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Conformal prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/calibration/uncertainty", response_model=CalibrationResponse)
async def uncertainty_estimation(
    request: UncertaintyRequest,
) -> CalibrationResponse:
    """Estimate predictive entropy and variance for a probability matrix."""
    try:
        estimator = BayesianUncertainty()
        probs = np.array(request.probs)
        data = {
            "entropy": estimator.predictive_entropy(probs).tolist(),
            "variance": estimator.predictive_variance(probs).tolist(),
            "mean_entropy": float(np.mean(estimator.predictive_entropy(probs))),
        }
        return CalibrationResponse(status="success", data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Uncertainty estimation failed")
        raise HTTPException(status_code=500, detail=str(exc))
