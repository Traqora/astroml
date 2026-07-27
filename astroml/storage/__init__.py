"""Artifact storage module with support for multiple backends."""
from astroml.storage.artifact_store import (
    ArtifactStore,
    GCSArtifactStore,
    LocalArtifactStore,
    S3ArtifactStore,
    create_artifact_store,
)
from astroml.storage.config import (
    ArtifactStorageConfig,
    GCSStorageConfig,
    LocalStorageConfig,
    S3StorageConfig,
)

__all__ = [
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "GCSArtifactStore",
    "create_artifact_store",
    "ArtifactStorageConfig",
    "LocalStorageConfig",
    "S3StorageConfig",
    "GCSStorageConfig",
]
