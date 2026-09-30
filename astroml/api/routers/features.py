from astroml.utils.exceptions import AstroMLError
"""Feature serving and management API router.

Exposes REST endpoints for online feature lookup, offline point-in-time historical
joins, feature registration, materialization, and statistical monitoring.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.features.feature_registry import FeatureStatus, FeatureType
from astroml.features.feature_store import get_feature_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/features", tags=["feature-store"])


class FeatureRegisterRequest(BaseModel):
    """Request model for registering a new feature."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique feature name")
    description: str = Field("", description="Feature description")
    feature_type: Literal["numeric", "categorical", "boolean", "text", "vector", "time_series"] = (
        "numeric"
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    owner: str = Field("system", description="Feature owner/team")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Computation parameters")


class FeatureRegisterResponse(BaseModel):
    """Response model for feature registration."""

    name: str
    feature_id: str
    status: str
    version: int


class OnlineFeatureRequest(BaseModel):
    """Request for real-time low-latency feature retrieval."""

    model_config = ConfigDict(extra="forbid")

    entity_ids: list[str] = Field(..., description="List of entity IDs to retrieve")
    feature_names: list[str] = Field(..., description="List of feature names")


class OnlineFeatureResponse(BaseModel):
    """Response containing online feature values."""

    features: dict[str, dict[str, Any]]
    entity_count: int


class HistoricalFeatureRequest(BaseModel):
    """Request for point-in-time historical feature join."""

    model_config = ConfigDict(extra="forbid")

    entities: list[dict[str, Any]] = Field(
        ..., description="List of entity records with timestamps"
    )
    feature_names: list[str] = Field(..., description="List of feature names to join")
    entity_col: str = "entity_id"
    timestamp_col: str = "timestamp"


class MaterializeRequest(BaseModel):
    """Request to materialize features from offline to online store."""

    model_config = ConfigDict(extra="forbid")

    feature_names: list[str]
    ttl_seconds: int | None = None


@router.post("/register", response_model=FeatureRegisterResponse)
async def register_feature(req: FeatureRegisterRequest) -> dict[str, Any]:
    """Register a new feature definition in the store."""
    try:
        fs = get_feature_store()
        feat_def = fs.register_feature(
            name=req.name,
            description=req.description,
            feature_type=FeatureType(req.feature_type),
            tags=req.tags,
            owner=req.owner,
            parameters=req.parameters,
        )
        return {
            "name": feat_def.name,
            "feature_id": feat_def.feature_id,
            "status": feat_def.status.value,
            "version": feat_def.version,
        }
    except Exception as exc:
        logger.error("Error registering feature %s: %s", req.name, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=list[dict[str, Any]])
async def list_features(
    status: str | None = Query(None, description="Filter by status"),
    owner: str | None = Query(None, description="Filter by owner"),
) -> list[dict[str, Any]]:
    """List registered feature definitions."""
    try:
        fs = get_feature_store()
        status_enum = FeatureStatus(status) if status else None
        features = fs.list_features(status=status_enum, owner=owner)
        return [f.to_dict() for f in features]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{name}", response_model=dict[str, Any])
async def get_feature_metadata(name: str) -> dict[str, Any]:
    """Get metadata for a specific feature."""
    fs = get_feature_store()
    feat_def = fs.storage.get_feature_definition(f"{name}_v1") or fs.storage.get_feature_definition(
        name
    )
    if not feat_def:
        raise HTTPException(status_code=404, detail=f"Feature '{name}' not found")
    return feat_def.to_dict()


@router.post("/online", response_model=OnlineFeatureResponse)
async def get_online_features(req: OnlineFeatureRequest) -> dict[str, Any]:
    """Low-latency online feature retrieval for real-time inference."""
    try:
        fs = get_feature_store()
        results = fs.get_online_features(
            entity_keys=req.entity_ids,
            feature_names=req.feature_names,
        )
        return {
            "features": results,
            "entity_count": len(req.entity_ids),
        }
    except Exception as exc:
        logger.error("Error in online feature lookup: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/historical", response_model=list[dict[str, Any]])
async def get_historical_features(req: HistoricalFeatureRequest) -> list[dict[str, Any]]:
    """Perform point-in-time correct join for historical training datasets."""
    try:
        fs = get_feature_store()
        entity_df = pd.DataFrame(req.entities)
        result_df = fs.get_historical_features(
            entity_df=entity_df,
            feature_names=req.feature_names,
            entity_col=req.entity_col,
            timestamp_col=req.timestamp_col,
        )
        # Convert timestamp to ISO string
        if req.timestamp_col in result_df.columns:
            result_df[req.timestamp_col] = result_df[req.timestamp_col].astype(str)
        return result_df.to_dict(orient="records")
    except Exception as exc:
        logger.error("Error in historical feature retrieval: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/materialize", response_model=dict[str, Any])
async def materialize_features(req: MaterializeRequest) -> dict[str, Any]:
    """Materialize batch features from offline storage to online store."""
    try:
        fs = get_feature_store()
        written = fs.materialize_to_online(
            feature_names=req.feature_names,
            ttl_seconds=req.ttl_seconds,
        )
        return {
            "status": "success",
            "features_materialized": req.feature_names,
            "values_written": written,
        }
    except Exception as exc:
        logger.error("Error in feature materialization: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{name}/stats", response_model=dict[str, Any])
async def get_feature_stats(name: str) -> dict[str, Any]:
    """Get statistical metrics and data drift monitoring summary for a feature."""
    try:
        fs = get_feature_store()
        stats = fs.get_feature_statistics(name)
        return stats
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
