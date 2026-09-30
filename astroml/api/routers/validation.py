from astroml.utils.exceptions import AstroMLError
"""FastAPI router for automated model validation and compliance verification."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.validation.compliance import ComplianceChecker
from astroml.validation.model_validator import ModelValidator
from astroml.validation.robustness import ModelRobustnessEvaluator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/validation", tags=["model-validation"])

_validator = ModelValidator()
_robustness_evaluator = ModelRobustnessEvaluator()
_compliance_checker = ComplianceChecker()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ValidateModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    weights: dict[str, Any] | None = None
    X_test: list[list[float]] = Field(..., min_length=1)
    y_test: list[float] = Field(..., min_length=1)
    sensitive_features: list[Any] | None = None
    baseline_metrics: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None
    required_compliance: list[str] | None = None


class RobustnessTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: dict[str, Any]
    X_test: list[list[float]] = Field(..., min_length=1)
    y_test: list[float] = Field(..., min_length=1)
    noise_levels: list[float] | None = None
    max_drop: float = 0.20


class ComplianceCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    metadata: dict[str, Any]
    metrics: dict[str, float] | None = None
    frameworks: list[str] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/gate", summary="Run pre-deployment CI/CD validation gate")
async def run_validation_gate(payload: ValidateModelRequest) -> dict[str, Any]:
    """Execute performance, fairness, robustness, compliance, and regression checks."""
    try:
        X = np.array(payload.X_test, dtype=np.float32)
        y = np.array(payload.y_test, dtype=np.float32)
        sens = (
            np.array(payload.sensitive_features)
            if payload.sensitive_features is not None
            else None
        )

        # Rebuild linear model or weights dict
        model_obj = None
        if payload.weights:
            model_obj = {k: np.array(v, dtype=np.float32) for k, v in payload.weights.items()}
        else:
            # Default mock linear predictor if weights omitted
            n_features = X.shape[1]
            model_obj = {
                "weight": np.ones((n_features, 1), dtype=np.float32),
                "bias": np.zeros(1, dtype=np.float32),
            }

        result = _validator.validate_model(
            model=model_obj,
            model_name=payload.model_name,
            version=payload.version,
            X_test=X,
            y_test=y,
            sensitive_features=sens,
            metadata=payload.metadata,
            baseline_metrics=payload.baseline_metrics,
            required_compliance_frameworks=payload.required_compliance,
        )

        return {
            "status": "success",
            "model_name": result.model_name,
            "version": result.version,
            "can_deploy": result.can_deploy,
            "overall_score": result.overall_score,
            "gate_decisions": result.gate_decisions,
            "blocking_reasons": result.blocking_reasons,
            "warnings": result.warnings,
            "performance_metrics": result.performance_metrics,
            "fairness_results": result.fairness_results,
            "robustness_results": result.robustness_results,
            "compliance_results": result.compliance_results,
        }
    except AstroMLError as e:
        logger.error("Validation gate evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/robustness", summary="Run isolated robustness and noise stress tests")
async def evaluate_robustness(payload: RobustnessTestRequest) -> dict[str, Any]:
    """Test model resilience against Gaussian/uniform noise, dropout, and FGSM."""
    try:
        X = np.array(payload.X_test, dtype=np.float32)
        y = np.array(payload.y_test, dtype=np.float32)
        model_obj = {k: np.array(v, dtype=np.float32) for k, v in payload.weights.items()}

        res = _robustness_evaluator.run_comprehensive_suite(
            model=model_obj,
            X=X,
            y=y,
            max_drop=payload.max_drop,
        )

        return {
            "status": "success",
            "robustness_score": res["robustness_score"],
            "all_passed": res["all_passed"],
            "passed_tests": res["passed_tests"],
            "total_tests": res["total_tests"],
            "results": res["results"],
        }
    except AstroMLError as e:
        logger.error("Robustness evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/compliance", summary="Run regulatory compliance audit")
async def evaluate_compliance(payload: ComplianceCheckRequest) -> dict[str, Any]:
    """Verify model compliance against EU AI Act, GDPR, SR 11-7, and Fair Lending."""
    try:
        report = _compliance_checker.evaluate_compliance(
            model_name=payload.model_name,
            version=payload.version,
            metadata=payload.metadata,
            validation_metrics=payload.metrics,
            frameworks=payload.frameworks,
        )

        return {
            "status": "success",
            "certificate_id": report.certificate_id,
            "model_name": report.model_name,
            "version": report.version,
            "overall_compliant": report.overall_compliant,
            "compliance_score": report.compliance_score,
            "framework_scores": report.framework_scores,
            "mandatory_violations": report.mandatory_violations,
            "summary": report.summary,
            "results": [
                {
                    "rule_id": r.rule_id,
                    "framework": r.framework,
                    "title": r.title,
                    "passed": r.passed,
                    "severity": r.severity.value,
                    "score": r.score,
                    "message": r.message,
                    "recommendation": r.recommendation,
                }
                for r in report.results
            ],
        }
    except AstroMLError as e:
        logger.error("Compliance evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/history", summary="List historical validation decisions")
async def list_validation_history() -> dict[str, Any]:
    """Retrieve history of all automated validation gate decisions."""
    history = _validator.get_validation_history()
    return {
        "status": "success",
        "total": len(history),
        "history": [h.to_dict() for h in history],
    }


@router.get("/history/{model_name}", summary="Get validation history for a model")
async def get_model_validation_history(model_name: str) -> dict[str, Any]:
    """Retrieve validation gate decisions for a specific model."""
    history = _validator.get_validation_history(model_name=model_name)
    return {
        "status": "success",
        "model_name": model_name,
        "total": len(history),
        "history": [h.to_dict() for h in history],
    }
