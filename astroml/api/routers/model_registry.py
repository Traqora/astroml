"""FastAPI router for Model Registry and Lineage Tracking endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.storage.model_store import ModelStore
from astroml.tracking.lineage import (
    DataLineageTracker,
    ModelLineage,
    TrainingLineage,
)
from astroml.tracking.metadata import ModelFramework, ModelMetadata, TaskType
from astroml.tracking.model_registry import (
    DeploymentEnvironment,
    ModelRegistry,
    ModelStage,
    SemanticVersion,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["model-registry"])

_model_store = ModelStore()
_lineage_tracker = DataLineageTracker()

# In-memory registry fallback / cache if DB session is in test mode or local
_in_memory_models: dict[str, dict[str, Any]] = {}
_in_memory_versions: dict[str, dict[str, dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RegisterModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    framework: str = Field(default="pytorch")
    task_type: str = Field(default="binary_classification")
    description: str = ""
    author: str = ""
    tags: list[str] = Field(default_factory=list)


class CreateVersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str | None = None
    artifact_uri: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    input_schema: dict[str, str] = Field(default_factory=dict)
    output_schema: dict[str, str] = Field(default_factory=dict)
    dataset_id: str | None = None
    dataset_version: str = "latest"
    commit_hash: str | None = None
    stage: str = Field(default="development", pattern=r"^(development|staging|production|archived)$")


class TransitionStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = Field(..., pattern=r"^(development|staging|production|archived)$")
    reason: str = ""


class UpdateMetricsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", summary="Register a new model entity")
async def register_model(payload: RegisterModelRequest) -> dict[str, Any]:
    """Register a new top-level model in the registry."""
    if payload.name in _in_memory_models:
        raise HTTPException(status_code=400, detail=f"Model '{payload.name}' already exists.")

    metadata = ModelMetadata(
        model_name=payload.name,
        framework=payload.framework,
        task_type=payload.task_type,
        description=payload.description,
        author=payload.author,
        tags=payload.tags,
    )

    _in_memory_models[payload.name] = {
        "name": payload.name,
        "framework": payload.framework,
        "task_type": payload.task_type,
        "description": payload.description,
        "author": payload.author,
        "tags": payload.tags,
        "metadata": metadata.to_dict(),
        "created_at": metadata.created_at,
        "is_active": True,
    }
    _in_memory_versions[payload.name] = {}

    return {
        "status": "success",
        "message": f"Model '{payload.name}' registered successfully.",
        "model": _in_memory_models[payload.name],
    }


@router.get("", summary="List registered models")
async def list_models(
    framework: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    is_active: bool | None = Query(default=None),
) -> dict[str, Any]:
    """List models with optional metadata filtering."""
    models = list(_in_memory_models.values())
    if framework:
        models = [m for m in models if m["framework"] == framework]
    if task_type:
        models = [m for m in models if m["task_type"] == task_type]
    if is_active is not None:
        models = [m for m in models if m["is_active"] == is_active]

    return {
        "status": "success",
        "total": len(models),
        "models": models,
    }


@router.get("/{model_name}", summary="Get model details")
async def get_model(model_name: str) -> dict[str, Any]:
    """Get metadata and overview for a specific model."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    versions = list(_in_memory_versions.get(model_name, {}).values())
    return {
        "status": "success",
        "model": _in_memory_models[model_name],
        "version_count": len(versions),
        "versions": [v["version"] for v in versions],
    }


@router.post("/{model_name}/versions", summary="Register a new model version")
async def create_model_version(model_name: str, payload: CreateVersionRequest) -> dict[str, Any]:
    """Register a new semantic version of a model with full lineage and metadata."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    # Determine semantic version
    existing_versions = _in_memory_versions[model_name]
    if payload.version:
        version_str = payload.version
        # Validate semver format
        try:
            SemanticVersion(version_str)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        if not existing_versions:
            version_str = "0.1.0"
        else:
            latest = max(SemanticVersion(v) for v in existing_versions.keys())
            version_str = f"{latest.major}.{latest.minor + 1}.0"

    if version_str in existing_versions:
        raise HTTPException(status_code=400, detail=f"Version '{version_str}' already exists for model '{model_name}'.")

    # Record Training Lineage
    lineage = TrainingLineage(
        dataset_id=payload.dataset_id or f"dataset_{model_name}",
        dataset_version=payload.dataset_version,
        commit_hash=payload.commit_hash,
        hyperparameters=payload.parameters,
    )
    _lineage_tracker.record_model_training(
        model_id=f"{model_name}:{version_str}",
        dataset_id=lineage.dataset_id,
        model_metadata={"version": version_str, "metrics": payload.metrics},
    )

    version_record = {
        "model_name": model_name,
        "version": version_str,
        "artifact_uri": payload.artifact_uri or f"./artifacts/models/{model_name}/{version_str}/model.pkl",
        "stage": payload.stage,
        "parameters": payload.parameters,
        "metrics": payload.metrics,
        "input_schema": payload.input_schema,
        "output_schema": payload.output_schema,
        "lineage": lineage.to_dict(),
        "created_at": lineage.created_at,
    }
    _in_memory_versions[model_name][version_str] = version_record

    return {
        "status": "success",
        "message": f"Version '{version_str}' created for model '{model_name}'.",
        "model_version": version_record,
    }


@router.get("/{model_name}/versions", summary="List all versions for a model")
async def list_model_versions(model_name: str) -> dict[str, Any]:
    """Retrieve all registered versions of a given model."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    versions = list(_in_memory_versions.get(model_name, {}).values())
    return {
        "status": "success",
        "model_name": model_name,
        "total_versions": len(versions),
        "versions": sorted(versions, key=lambda v: SemanticVersion(v["version"]), reverse=True),
    }


@router.get("/{model_name}/versions/{version}", summary="Get model version details")
async def get_model_version(model_name: str, version: str) -> dict[str, Any]:
    """Retrieve detailed metadata and lineage for a specific model version."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    versions = _in_memory_versions.get(model_name, {})
    if version not in versions:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found for model '{model_name}'.")

    return {
        "status": "success",
        "model_version": versions[version],
    }


@router.post("/{model_name}/versions/{version}/stage", summary="Transition model version stage")
async def transition_version_stage(
    model_name: str, version: str, payload: TransitionStageRequest
) -> dict[str, Any]:
    """Promote or demote a model version stage (development, staging, production, archived)."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    versions = _in_memory_versions.get(model_name, {})
    if version not in versions:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    # If transitioning to production, demote previous production version to archived/staging
    if payload.stage == ModelStage.PRODUCTION.value:
        for v_name, v_data in versions.items():
            if v_data["stage"] == ModelStage.PRODUCTION.value and v_name != version:
                v_data["stage"] = ModelStage.ARCHIVED.value
                logger.info("Demoted previous production model %s:%s to archived", model_name, v_name)

    versions[version]["stage"] = payload.stage
    versions[version]["stage_transition_reason"] = payload.reason

    return {
        "status": "success",
        "message": f"Model '{model_name}' version '{version}' transitioned to '{payload.stage}'.",
        "stage": payload.stage,
    }


@router.get("/{model_name}/production", summary="Get active production model version")
async def get_production_version(model_name: str) -> dict[str, Any]:
    """Retrieve the current production version for a model."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    versions = _in_memory_versions.get(model_name, {})
    prod_version = next((v for v in versions.values() if v["stage"] == ModelStage.PRODUCTION.value), None)

    if not prod_version:
        raise HTTPException(status_code=404, detail=f"No production version found for model '{model_name}'.")

    return {
        "status": "success",
        "model_name": model_name,
        "production_version": prod_version,
    }


@router.post("/{model_name}/versions/{version}/metrics", summary="Update version evaluation metrics")
async def update_version_metrics(
    model_name: str, version: str, payload: UpdateMetricsRequest
) -> dict[str, Any]:
    """Update or append metrics for a specific model version."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    versions = _in_memory_versions.get(model_name, {})
    if version not in versions:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    versions[version]["metrics"].update(payload.metrics)
    return {
        "status": "success",
        "message": "Metrics updated successfully.",
        "metrics": versions[version]["metrics"],
    }


@router.get("/{model_name}/versions/{version}/lineage", summary="Get model lineage graph")
async def get_model_lineage(model_name: str, version: str) -> dict[str, Any]:
    """Retrieve upstream dataset, code, hyperparameters, and downstream lineage graph."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    versions = _in_memory_versions.get(model_name, {})
    if version not in versions:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    ver_data = versions[version]
    model_id = f"{model_name}:{version}"
    lineage_info = _lineage_tracker.get_lineage(model_id)

    return {
        "status": "success",
        "model_name": model_name,
        "version": version,
        "training_lineage": ver_data.get("lineage", {}),
        "graph": lineage_info,
    }


@router.delete("/{model_name}/versions/{version}", summary="Delete a model version")
async def delete_model_version(model_name: str, version: str) -> dict[str, Any]:
    """Delete a model version and remove its stored artifacts."""
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    versions = _in_memory_versions.get(model_name, {})
    if version not in versions:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    del versions[version]
    _model_store.delete_version_artifacts(model_name, version)

    return {
        "status": "success",
        "message": f"Version '{version}' of model '{model_name}' deleted.",
    }


@router.get(
    "/{model_name}/versions/{version}/mlflow-run",
    summary="Get linked MLflow run details",
)
async def get_mlflow_run_details(model_name: str, version: str) -> dict[str, Any]:
    """Retrieve the MLflow run details linked to a model version.

    Returns run metadata (metrics, params, tags, artifact URI) from MLflow
    for the run stored on this registry version via ``mlflow_run_id``.
    """
    if model_name not in _in_memory_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    versions = _in_memory_versions.get(model_name, {})
    if version not in versions:
        raise HTTPException(status_code=404, detail=f"Version '{version}' not found.")

    ver_data = versions[version]
    mlflow_run_id = ver_data.get("mlflow_run_id")

    if not mlflow_run_id:
        raise HTTPException(
            status_code=404,
            detail=f"Version '{version}' of model '{model_name}' has no linked MLflow run.",
        )

    try:
        import mlflow

        run = mlflow.get_run(mlflow_run_id)
        return {
            "status": "success",
            "model_name": model_name,
            "version": version,
            "mlflow_run": {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags),
                "artifact_uri": run.info.artifact_uri,
            },
        }
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlflow package is not installed. Install it with: pip install mlflow",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch MLflow run {mlflow_run_id}: {exc}",
        )
