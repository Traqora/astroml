"""Validation router for issue #303.

Provides endpoints for:
- Validating transaction data
- Validating account data
- Validating feature data
- Running validation pipelines
- Getting validation metrics
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from astroml.validation.api_validation import (
    ValidationResult,
    validate_transaction_input,
    validate_account_input,
    validate_score_request,
    validate_feature_data,
    CustomValidationRules,
)
from astroml.validation.pipeline import (
    ValidationPipeline,
    ValidationStage,
    ValidationStageConfig,
    PipelineResult,
    create_transaction_pipeline,
    create_account_pipeline,
    create_feature_pipeline,
)

router = APIRouter(prefix="/validation", tags=["validation"])


# ─── Request/Response Schemas ─────────────────────────────────────────────

class ValidationRequest(BaseModel):
    """Request schema for validation endpoint."""

    data: Dict[str, Any] = Field(..., description="Data to validate")
    data_type: str = Field(default="transaction", description="Type of data: transaction, account, feature")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Validation context")


class PipelineRequest(BaseModel):
    """Request schema for pipeline validation."""

    data: Dict[str, Any] = Field(..., description="Data to validate")
    pipeline_type: str = Field(default="transaction", description="Pipeline type: transaction, account, feature")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Validation context")


class MetricsResponse(BaseModel):
    """Response schema for validation metrics."""

    total_runs: int
    successful_runs: int
    failed_runs: int
    average_execution_time_ms: float


# ─── Validation Endpoints ─────────────────────────────────────────────────

@router.post("/validate", response_model=Dict[str, Any])
async def validate_data(request: ValidationRequest):
    """Validate data using appropriate schema validation."""
    data_type = request.data_type.lower()
    context = request.context or {}

    if data_type == "transaction":
        result = validate_transaction_input(request.data)
    elif data_type == "account":
        result = validate_account_input(request.data)
    elif data_type == "feature":
        result = validate_feature_data(request.data)
    elif data_type == "score_request":
        result = validate_score_request(request.data)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")

    return result.to_dict()


@router.post("/pipeline", response_model=Dict[str, Any])
async def run_validation_pipeline(request: PipelineRequest):
    """Run data through validation pipeline."""
    pipeline_type = request.pipeline_type.lower()
    context = request.context or {}

    if pipeline_type == "transaction":
        pipeline = create_transaction_pipeline()
        context["data_type"] = "transaction"
    elif pipeline_type == "account":
        pipeline = create_account_pipeline()
        context["data_type"] = "account"
    elif pipeline_type == "feature":
        pipeline = create_feature_pipeline()
        context["data_type"] = "feature"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline type: {pipeline_type}")

    result = pipeline.run(request.data, context)
    return result.to_dict()


@router.get("/metrics", response_model=MetricsResponse)
async def get_validation_metrics():
    """Get validation pipeline metrics."""
    # Create a temporary pipeline to get metrics
    pipeline = create_transaction_pipeline()
    metrics = pipeline.get_metrics()
    return MetricsResponse(**metrics)


@router.post("/metrics/reset")
async def reset_validation_metrics():
    """Reset validation pipeline metrics."""
    pipeline = create_transaction_pipeline()
    pipeline.reset_metrics()
    return {"message": "Metrics reset successfully"}


@router.post("/custom/transaction-limits")
async def validate_transaction_limits(
    amount: float,
    max_amount: float = 1_000_000_000,
):
    """Validate transaction amount against limits."""
    result = CustomValidationRules.validate_transaction_limits(amount, max_amount)
    return result.to_dict()


@router.post("/custom/account-age")
async def validate_account_age(
    created_at: str,
    min_age_days: int = 0,
):
    """Validate account age."""
    from datetime import datetime

    try:
        created_dt = datetime.fromisoformat(created_at)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format")

    result = CustomValidationRules.validate_account_age(created_dt, min_age_days)
    return result.to_dict()


@router.post("/custom/feature-ranges")
async def validate_feature_ranges(
    features: Dict[str, float],
    ranges: Dict[str, tuple],
):
    """Validate feature values against expected ranges."""
    result = CustomValidationRules.validate_feature_range(features, ranges)
    return result.to_dict()


@router.post("/custom/no-negative")
async def validate_no_negative_features(features: Dict[str, Any]):
    """Ensure no features have negative values."""
    result = CustomValidationRules.validate_no_negative_features(features)
    return result.to_dict()
