"""Configurable artifact store with fsspec support for S3, GCS, and local storage."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

import fsspec
from fsspec.spec import AbstractFileSystem

logger = logging.getLogger(__name__)


class ArtifactStore(ABC):
    """Abstract base class for artifact storage backends."""

    @abstractmethod
    def save(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Save a local file to the artifact store.

        Args:
            local_path: Path to local file to save
            remote_path: Destination path in artifact store

        Returns:
            Full URI of saved artifact
        """
        pass

    @abstractmethod
    def load(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """Load an artifact from the store to local filesystem.

        Args:
            remote_path: Path in artifact store
            local_path: Destination path on local filesystem

        Returns:
            Path to loaded file
        """
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if artifact exists in store.

        Args:
            remote_path: Path in artifact store

        Returns:
            True if artifact exists
        """
        pass

    @abstractmethod
    def delete(self, remote_path: str) -> None:
        """Delete an artifact from the store.

        Args:
            remote_path: Path in artifact store
        """
        pass

    @abstractmethod
    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List artifacts in the store.

        Args:
            prefix: Optional prefix to filter artifacts

        Returns:
            List of artifact paths
        """
        pass

    @abstractmethod
    def get_uri(self, remote_path: str) -> str:
        """Get the full URI for an artifact.

        Args:
            remote_path: Path in artifact store

        Returns:
            Full URI (e.g., s3://bucket/path, gs://bucket/path, file:///path)
        """
        pass


class LocalArtifactStore(ArtifactStore):
    """Local filesystem artifact store."""

    def __init__(self, base_path: Union[str, Path]):
        """Initialize local artifact store.

        Args:
            base_path: Base directory for artifacts
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.fs: AbstractFileSystem = fsspec.filesystem("file")
        logger.info(f"Initialized local artifact store at {self.base_path}")

    def save(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Save a local file to the artifact store."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        dest_path = self.base_path / remote_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        self.fs.copy(str(local_path), str(dest_path), recursive=False)
        logger.info(f"Saved artifact: {local_path} -> {dest_path}")
        return self.get_uri(remote_path)

    def load(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """Load an artifact from the store to local filesystem."""
        local_path = Path(local_path)
        src_path = self.base_path / remote_path

        if not src_path.exists():
            raise FileNotFoundError(f"Artifact not found: {src_path}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.fs.copy(str(src_path), str(local_path), recursive=False)
        logger.info(f"Loaded artifact: {src_path} -> {local_path}")
        return local_path

    def exists(self, remote_path: str) -> bool:
        """Check if artifact exists in store."""
        return (self.base_path / remote_path).exists()

    def delete(self, remote_path: str) -> None:
        """Delete an artifact from the store."""
        path = self.base_path / remote_path
        if path.exists():
            self.fs.rm(str(path), recursive=False)
            logger.info(f"Deleted artifact: {path}")

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List artifacts in the store."""
        search_path = self.base_path / prefix if prefix else self.base_path
        if not search_path.exists():
            return []

        artifacts = []
        for root, dirs, files in os.walk(search_path):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(self.base_path)
                artifacts.append(str(rel_path))
        return artifacts

    def get_uri(self, remote_path: str) -> str:
        """Get the full URI for an artifact."""
        full_path = self.base_path / remote_path
        return f"file://{full_path.absolute()}"


class S3ArtifactStore(ArtifactStore):
    """AWS S3 artifact store."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """Initialize S3 artifact store.

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix for all artifacts
            aws_access_key_id: AWS access key (uses env var if not provided)
            aws_secret_access_key: AWS secret key (uses env var if not provided)
            region_name: AWS region (uses env var or default if not provided)
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        
        # Prepare S3 credentials
        s3_kwargs = {}
        if aws_access_key_id:
            s3_kwargs["key"] = aws_access_key_id
        if aws_secret_access_key:
            s3_kwargs["secret"] = aws_secret_access_key
        if region_name:
            s3_kwargs["client_kwargs"] = {"region_name": region_name}

        self.fs: AbstractFileSystem = fsspec.filesystem("s3", **s3_kwargs)
        logger.info(f"Initialized S3 artifact store: s3://{bucket}/{prefix}")

    def _get_s3_path(self, remote_path: str) -> str:
        """Get full S3 path with bucket and prefix."""
        if self.prefix:
            return f"{self.bucket}/{self.prefix}/{remote_path}".lstrip("/")
        return f"{self.bucket}/{remote_path}".lstrip("/")

    def save(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Save a local file to S3."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        s3_path = self._get_s3_path(remote_path)
        self.fs.put(str(local_path), s3_path)
        logger.info(f"Saved artifact to S3: {local_path} -> s3://{s3_path}")
        return self.get_uri(remote_path)

    def load(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """Load an artifact from S3 to local filesystem."""
        local_path = Path(local_path)
        s3_path = self._get_s3_path(remote_path)

        if not self.exists(remote_path):
            raise FileNotFoundError(f"Artifact not found in S3: s3://{s3_path}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get(s3_path, str(local_path))
        logger.info(f"Loaded artifact from S3: s3://{s3_path} -> {local_path}")
        return local_path

    def exists(self, remote_path: str) -> bool:
        """Check if artifact exists in S3."""
        s3_path = self._get_s3_path(remote_path)
        return self.fs.exists(s3_path)

    def delete(self, remote_path: str) -> None:
        """Delete an artifact from S3."""
        s3_path = self._get_s3_path(remote_path)
        if self.exists(remote_path):
            self.fs.rm(s3_path)
            logger.info(f"Deleted artifact from S3: s3://{s3_path}")

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List artifacts in S3."""
        search_prefix = self.prefix
        if prefix:
            search_prefix = f"{self.prefix}/{prefix}".lstrip("/")

        s3_prefix = f"{self.bucket}/{search_prefix}".lstrip("/")
        try:
            files = self.fs.ls(s3_prefix, detail=False)
            # Remove bucket and prefix from paths
            artifacts = []
            for f in files:
                # Extract relative path
                rel_path = f.replace(f"{self.bucket}/", "")
                if self.prefix:
                    rel_path = rel_path.replace(f"{self.prefix}/", "")
                artifacts.append(rel_path)
            return artifacts
        except FileNotFoundError:
            return []

    def get_uri(self, remote_path: str) -> str:
        """Get the full S3 URI for an artifact."""
        s3_path = self._get_s3_path(remote_path)
        return f"s3://{s3_path}"


class GCSArtifactStore(ArtifactStore):
    """Google Cloud Storage artifact store."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        """Initialize GCS artifact store.

        Args:
            bucket: GCS bucket name
            prefix: Optional prefix for all artifacts
            project_id: GCP project ID (uses env var if not provided)
            credentials_path: Path to service account JSON (uses env var if not provided)
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

        # Prepare GCS credentials
        gcs_kwargs = {}
        if project_id:
            gcs_kwargs["project"] = project_id
        if credentials_path:
            gcs_kwargs["token"] = credentials_path

        self.fs: AbstractFileSystem = fsspec.filesystem("gs", **gcs_kwargs)
        logger.info(f"Initialized GCS artifact store: gs://{bucket}/{prefix}")

    def _get_gcs_path(self, remote_path: str) -> str:
        """Get full GCS path with bucket and prefix."""
        if self.prefix:
            return f"{self.bucket}/{self.prefix}/{remote_path}".lstrip("/")
        return f"{self.bucket}/{remote_path}".lstrip("/")

    def save(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Save a local file to GCS."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        gcs_path = self._get_gcs_path(remote_path)
        self.fs.put(str(local_path), gcs_path)
        logger.info(f"Saved artifact to GCS: {local_path} -> gs://{gcs_path}")
        return self.get_uri(remote_path)

    def load(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """Load an artifact from GCS to local filesystem."""
        local_path = Path(local_path)
        gcs_path = self._get_gcs_path(remote_path)

        if not self.exists(remote_path):
            raise FileNotFoundError(f"Artifact not found in GCS: gs://{gcs_path}")

        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.fs.get(gcs_path, str(local_path))
        logger.info(f"Loaded artifact from GCS: gs://{gcs_path} -> {local_path}")
        return local_path

    def exists(self, remote_path: str) -> bool:
        """Check if artifact exists in GCS."""
        gcs_path = self._get_gcs_path(remote_path)
        return self.fs.exists(gcs_path)

    def delete(self, remote_path: str) -> None:
        """Delete an artifact from GCS."""
        gcs_path = self._get_gcs_path(remote_path)
        if self.exists(remote_path):
            self.fs.rm(gcs_path)
            logger.info(f"Deleted artifact from GCS: gs://{gcs_path}")

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List artifacts in GCS."""
        search_prefix = self.prefix
        if prefix:
            search_prefix = f"{self.prefix}/{prefix}".lstrip("/")

        gcs_prefix = f"{self.bucket}/{search_prefix}".lstrip("/")
        try:
            files = self.fs.ls(gcs_prefix, detail=False)
            # Remove bucket and prefix from paths
            artifacts = []
            for f in files:
                rel_path = f.replace(f"{self.bucket}/", "")
                if self.prefix:
                    rel_path = rel_path.replace(f"{self.prefix}/", "")
                artifacts.append(rel_path)
            return artifacts
        except FileNotFoundError:
            return []

    def get_uri(self, remote_path: str) -> str:
        """Get the full GCS URI for an artifact."""
        gcs_path = self._get_gcs_path(remote_path)
        return f"gs://{gcs_path}"


def create_artifact_store(artifact_uri: str, **kwargs) -> ArtifactStore:
    """Factory function to create artifact store from URI.

    Args:
        artifact_uri: URI specifying storage backend
            - "file:///path/to/artifacts" for local storage
            - "s3://bucket/prefix" for S3
            - "gs://bucket/prefix" for GCS
        **kwargs: Additional arguments passed to store constructor

    Returns:
        Configured ArtifactStore instance

    Raises:
        ValueError: If URI scheme is not supported
    """
    if artifact_uri.startswith("file://"):
        path = artifact_uri.replace("file://", "")
        return LocalArtifactStore(path)
    elif artifact_uri.startswith("s3://"):
        parts = artifact_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3ArtifactStore(bucket, prefix, **kwargs)
    elif artifact_uri.startswith("gs://"):
        parts = artifact_uri.replace("gs://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return GCSArtifactStore(bucket, prefix, **kwargs)
    else:
        raise ValueError(
            f"Unsupported artifact URI scheme: {artifact_uri}. "
            "Supported schemes: file://, s3://, gs://"
        )
