"""Pydantic schemas for model registry."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Model registry schemas
class ModelRegistryIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    version: str = Field(..., min_length=1, max_length=64)
    path: str = Field(..., min_length=1)
    owner: Optional[str] = Field(None, max_length=128)
    tags: Optional[List[str]] = None
    mlflow_run_id: Optional[str] = Field(None, max_length=128)
    metrics: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(default="inactive", pattern="^(inactive|active|deprecated)$")


class ModelRegistryUpdateIn(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    version: Optional[str] = Field(None, min_length=1, max_length=64)
    path: Optional[str] = Field(None, min_length=1)
    owner: Optional[str] = Field(None, max_length=128)
    tags: Optional[List[str]] = None
    mlflow_run_id: Optional[str] = Field(None, max_length=128)
    metrics: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern="^(inactive|active|deprecated)$")


class ModelRegistryOut(BaseModel):
    id: int
    name: str
    version: str
    path: str
    owner: Optional[str]
    tags: Optional[List[str]]
    mlflow_run_id: Optional[str]
    metrics: Optional[Dict[str, Any]]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    data: List[ModelRegistryOut]
    page: int
    page_size: int
    total: int


class ModelVersionTransitionIn(BaseModel):
    target_status: str = Field(..., pattern="^(inactive|active|deprecated)$")


class ModelComparisonIn(BaseModel):
    model_ids: List[int] = Field(..., min_length=2)


class ModelComparisonOut(BaseModel):
    models: List[ModelRegistryOut]
    comparison: Dict[str, Any]


class ModelSearchIn(BaseModel):
    query: str
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class ModelTagsUpdateIn(BaseModel):
    add_tags: Optional[List[str]] = None
    remove_tags: Optional[List[str]] = None

# Add to existing schemas/model_registry.py

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class SemanticVersion(BaseModel):
    """Semantic version model."""
    major: int = Field(ge=0)
    minor: int = Field(ge=0)
    patch: int = Field(ge=0)

    @validator('patch')
    def validate_patch(cls, v):
        if v < 0:
            raise ValueError('Patch version must be >= 0')
        return v

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


class ModelVersionCreate(BaseModel):
    """Create a new model version."""
    version: Optional[str] = None
    artifact_path: str
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    status: Optional[str] = "training"
    metadata: Optional[Dict[str, Any]] = None
    auto_version: bool = True


class ModelVersionUpdate(BaseModel):
    """Update a model version."""
    status: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelVersionTransition(BaseModel):
    """Transition a model version to a new status."""
    target_status: str
    reason: Optional[str] = None


class RollbackRequest(BaseModel):
    """Rollback request."""
    target_version: str
    reason: str


class ABTestCreate(BaseModel):
    """Create an A/B test."""
    control_version: str
    treatment_version: str
    traffic_split: float = Field(ge=0.0, le=1.0)
    metrics: Optional[List[str]] = None


class DeploymentRequest(BaseModel):
    """Deploy a model version."""
    version_id: int
    environment: str
    deployed_by: Optional[str] = None
    notes: Optional[str] = None


class VersionComparisonResult(BaseModel):
    """Version comparison result."""
    versions: List[Dict[str, Any]]
    metrics: Dict[str, Dict[str, Optional[float]]]
    summary: Dict[str, Dict[str, Any]]


class VersionHistoryItem(BaseModel):
    """Version history item."""
    id: int
    version: str
    status: str
    metrics: Optional[Dict[str, Any]]
    created_at: str
    deployed_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

