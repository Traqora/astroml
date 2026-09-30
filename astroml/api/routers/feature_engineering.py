"""Automated Feature Engineering API Router for AstroML.

Provides endpoints for automated relational feature synthesis with Featuretools,
feature pruning, importance ranking, and primitives discovery.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

try:
    import featuretools as ft

    _HAS_FT = True
except ImportError:
    _HAS_FT = False

from astroml.features.deep_feature_synthesis import (
    DeepFeatureSynthesizer,
    DFSPipeline,
    prune_features,
    rank_feature_importance,
)
from astroml.preprocessing.auto_feature_engineering import build_transaction_entityset

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/feature-engineering", tags=["feature-engineering"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FeatureSynthesisRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(..., description="List of transaction records")
    accounts: list[dict[str, Any]] | None = Field(
        default=None, description="Optional account records"
    )
    max_depth: int = Field(default=2, ge=1, le=3)
    target_entity: str = Field(default="accounts")


class FeatureSynthesisResponse(BaseModel):
    num_features: int
    feature_names: list[str]
    sample_records: list[dict[str, Any]]


class FeatureSelectionRequest(BaseModel):
    feature_data: list[dict[str, Any]] = Field(..., description="Feature matrix as list of records")
    variance_threshold: float = Field(default=0.01, ge=0.0)
    correlation_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    max_missing_rate: float = Field(default=0.5, ge=0.0, le=1.0)


class FeatureSelectionResponse(BaseModel):
    retained_features: list[str]
    num_retained: int
    pruned_records: list[dict[str, Any]]


class FeatureImportanceRequest(BaseModel):
    feature_data: list[dict[str, Any]]
    target: list[int | float]
    method: str = Field(
        default="random_forest",
        description="'random_forest', 'gradient_boosting', 'mutual_info', 'correlation'",
    )
    top_k: int | None = Field(default=20, ge=1)


class FeatureImportanceResponse(BaseModel):
    rankings: list[dict[str, Any]]


class PrimitivesResponse(BaseModel):
    aggregation_primitives: list[str]
    transform_primitives: list[str]


class AutoPipelineRequest(BaseModel):
    transactions: list[dict[str, Any]]
    accounts: list[dict[str, Any]] | None = None
    target_labels: dict[str, int] | None = None
    max_depth: int = Field(default=2, ge=1, le=3)
    top_k: int = Field(default=20, ge=1)


class AutoPipelineResponse(BaseModel):
    status: str
    num_features_generated: int
    num_features_selected: int
    top_features: list[str]
    feature_matrix_sample: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict[str, str]:
    """Health check for automated feature engineering service."""
    return {"status": "ok", "service": "feature-engineering"}


@router.get("/primitives", response_model=PrimitivesResponse)
async def get_primitives() -> PrimitivesResponse:
    """List available aggregation and transformation primitives."""
    if not _HAS_FT or ft is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Featuretools library is not installed",
        )
    aggs = list(ft.primitives.get_aggregation_primitives().keys())
    trans = list(ft.primitives.get_transform_primitives().keys())
    return PrimitivesResponse(aggregation_primitives=aggs, transform_primitives=trans)


@router.post("/synthesize", response_model=FeatureSynthesisResponse)
async def synthesize_features(request: FeatureSynthesisRequest) -> FeatureSynthesisResponse:
    """Execute Deep Feature Synthesis on provided relational data."""
    if not request.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction dataset cannot be empty",
        )

    tx_df = pd.DataFrame(request.transactions)
    acc_df = pd.DataFrame(request.accounts) if request.accounts else None

    es = build_transaction_entityset(transactions_df=tx_df, accounts_df=acc_df)
    dfs = DeepFeatureSynthesizer(
        target_dataframe_name=request.target_entity,
        max_depth=request.max_depth,
    )
    feature_matrix, _ = dfs.fit_transform(es)

    sample = feature_matrix.head(10).reset_index().to_dict(orient="records")
    return FeatureSynthesisResponse(
        num_features=len(feature_matrix.columns),
        feature_names=list(feature_matrix.columns),
        sample_records=sample,
    )


@router.post("/select-features", response_model=FeatureSelectionResponse)
async def select_features_endpoint(request: FeatureSelectionRequest) -> FeatureSelectionResponse:
    """Filter features by variance, missingness, and correlation."""
    if not request.feature_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feature data cannot be empty",
        )

    df = pd.DataFrame(request.feature_data)
    pruned_df, retained_cols = prune_features(
        feature_matrix=df,
        variance_threshold=request.variance_threshold,
        correlation_threshold=request.correlation_threshold,
        max_missing_rate=request.max_missing_rate,
    )

    return FeatureSelectionResponse(
        retained_features=retained_cols,
        num_retained=len(retained_cols),
        pruned_records=pruned_df.head(10).to_dict(orient="records"),
    )


@router.post("/rank-importance", response_model=FeatureImportanceResponse)
async def rank_importance_endpoint(request: FeatureImportanceRequest) -> FeatureImportanceResponse:
    """Rank features by importance against target labels."""
    if not request.feature_data or not request.target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feature data and target cannot be empty",
        )

    df = pd.DataFrame(request.feature_data)
    if len(df) != len(request.target):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Length mismatch: feature_data ({len(df)}) != target ({len(request.target)})",
        )

    rankings_df = rank_feature_importance(
        feature_matrix=df,
        target=np.array(request.target),
        method=request.method,
        top_k=request.top_k,
    )

    return FeatureImportanceResponse(rankings=rankings_df.to_dict(orient="records"))


@router.post("/pipeline", response_model=AutoPipelineResponse)
async def run_pipeline_endpoint(request: AutoPipelineRequest) -> AutoPipelineResponse:
    """Run end-to-end automated feature engineering pipeline."""
    if not request.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction dataset cannot be empty",
        )

    tx_df = pd.DataFrame(request.transactions)
    acc_df = pd.DataFrame(request.accounts) if request.accounts else None

    es = build_transaction_entityset(transactions_df=tx_df, accounts_df=acc_df)

    target_series = None
    df_names = [df.ww.name for df in es.dataframes]
    if request.target_labels and "accounts" in df_names:
        account_ids = es["accounts"]["account_id"].tolist()
        target_series = pd.Series([request.target_labels.get(acc, 0) for acc in account_ids])

    pipeline = DFSPipeline(
        max_depth=request.max_depth,
        top_k_features=request.top_k,
    )
    final_matrix = pipeline.fit_transform(es, target=target_series)

    return AutoPipelineResponse(
        status="success",
        num_features_generated=len(pipeline.synthesizer.feature_defs_ or []),
        num_features_selected=len(pipeline.selected_features_),
        top_features=pipeline.selected_features_[: request.top_k],
        feature_matrix_sample=final_matrix.head(10).reset_index().to_dict(orient="records"),
    )
