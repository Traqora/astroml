"""Configuration for artifact storage."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class S3StorageConfig(BaseModel):
    """S3 storage configuration."""

    bucket: str = Field(..., description="S3 bucket name")
    prefix: str = Field(default="", description="Prefix for all artifacts in bucket")
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key (uses env var if not provided)"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret key (uses env var if not provided)"
    )
    region_name: Optional[str] = Field(
        default=None, description="AWS region (uses default if not provided)"
    )


class GCSStorageConfig(BaseModel):
    """Google Cloud Storage configuration."""

    bucket: str = Field(..., description="GCS bucket name")
    prefix: str = Field(default="", description="Prefix for all artifacts in bucket")
    project_id: Optional[str] = Field(
        default=None, description="GCP project ID (uses env var if not provided)"
    )
    credentials_path: Optional[str] = Field(
        default=None, description="Path to service account JSON (uses env var if not provided)"
    )


class LocalStorageConfig(BaseModel):
    """Local filesystem storage configuration."""

    path: str = Field(default="artifacts", description="Base directory for artifacts")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path is valid."""
        if not v:
            raise ValueError("Storage path cannot be empty")
        return v


class ArtifactStorageConfig(BaseModel):
    """Main artifact storage configuration."""

    backend: Literal["local", "s3", "gcs"] = Field(
        default="local", description="Storage backend to use"
    )
    local: LocalStorageConfig = Field(
        default_factory=LocalStorageConfig, description="Local storage config"
    )
    s3: S3StorageConfig = Field(
        default_factory=lambda: S3StorageConfig(bucket=""),
        description="S3 storage config",
    )
    gcs: GCSStorageConfig = Field(
        default_factory=lambda: GCSStorageConfig(bucket=""),
        description="GCS storage config",
    )

    def get_artifact_uri(self) -> str:
        """Get artifact URI based on configured backend.

        Returns:
            URI string (e.g., "file:///path", "s3://bucket/prefix", "gs://bucket/prefix")
        """
        if self.backend == "local":
            return f"file://{Path(self.local.path).absolute()}"
        elif self.backend == "s3":
            uri = f"s3://{self.s3.bucket}"
            if self.s3.prefix:
                uri += f"/{self.s3.prefix}"
            return uri
        elif self.backend == "gcs":
            uri = f"gs://{self.gcs.bucket}"
            if self.gcs.prefix:
                uri += f"/{self.gcs.prefix}"
            return uri
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> ArtifactStorageConfig:
        """Create config from dictionary."""
        return cls(**data)
