"""Artifact storage management using fsspec for multi-backend support.

This module provides a unified interface for saving and loading artifacts
(models, checkpoints, results) to various storage backends including local
filesystem, AWS S3, and Google Cloud Storage.

Supported URIs:
- Local filesystem: /path/to/artifacts or ./artifacts
- AWS S3: s3://bucket-name/path
- Google Cloud Storage: gs://bucket-name/path
- HTTP/HTTPS: https://example.com/artifacts (read-only)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import fsspec
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Unified artifact storage using fsspec for multi-backend support."""
    
    def __init__(self, artifact_uri: str | None = None):
        """Initialize artifact store with optional URI override.
        
        Args:
            artifact_uri: Storage URI (e.g., 's3://bucket/path', 'gs://bucket/path', '/local/path').
                         If None, defaults to local './artifacts' directory.
                         Can also be set via ASTROML_ARTIFACT_URI environment variable.
        """
        # Determine artifact URI from parameter, environment, or default
        self.artifact_uri = (
            artifact_uri 
            or os.environ.get('ASTROML_ARTIFACT_URI', './artifacts')
        )
        
        # Normalize local paths
        if not self.artifact_uri.startswith(('s3://', 'gs://', 'http://', 'http://')):
            self.artifact_uri = str(Path(self.artifact_uri).resolve())
        
        logger.info(f"Initialized artifact store: {self.artifact_uri}")
        
        # Initialize filesystem
        self.fs = fsspec.filesystem(self._get_protocol())
        
        # Create directory if it's local filesystem
        if self._get_protocol() == 'file':
            os.makedirs(self.artifact_uri, exist_ok=True)
    
    def _get_protocol(self) -> str:
        """Extract protocol from URI."""
        if self.artifact_uri.startswith('s3://'):
            return 's3'
        elif self.artifact_uri.startswith('gs://'):
            return 'gs'
        elif self.artifact_uri.startswith('http://'):
            return 'http'
        elif self.artifact_uri.startswith('https://'):
            return 'https'
        else:
            return 'file'
    
    def _normalize_path(self, relative_path: str) -> str:
        """Normalize and combine artifact URI with relative path."""
        # Remove leading slashes from relative path
        relative_path = relative_path.lstrip('/')
        
        # Use appropriate separator for protocol
        if self._get_protocol() == 'file':
            return os.path.join(self.artifact_uri, relative_path)
        else:
            # For cloud storage, use forward slashes
            return f"{self.artifact_uri.rstrip('/')}/{relative_path}"
    
    def save_model(
        self,
        model: nn.Module | Dict[str, Any],
        path: str,
        metadata: Dict[str, Any] | None = None
    ) -> str:
        """Save model to artifact store.
        
        Args:
            model: PyTorch model or state dict to save
            path: Relative path within artifact store
            metadata: Optional metadata to save alongside model
            
        Returns:
            Full artifact URI of saved model
        """
        full_path = self._normalize_path(path)
        
        try:
            # Get state dict if model is nn.Module
            if isinstance(model, nn.Module):
                state_dict = model.state_dict()
            else:
                state_dict = model
            
            # For local filesystem, use direct torch.save
            if self._get_protocol() == 'file':
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                torch.save(state_dict, full_path)
            else:
                # For cloud storage, save to temporary location and upload
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp_path = tmp.name
                    torch.save(state_dict, tmp_path)
                
                try:
                    with open(tmp_path, 'rb') as f:
                        self.fs.upload(tmp_path, full_path)
                finally:
                    os.remove(tmp_path)
            
            logger.info(f"Saved model to {full_path}")
            
            # Save metadata if provided
            if metadata:
                self.save_metadata(metadata, path.replace('.pt', '_metadata.json'))
            
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to save model to {full_path}: {e}")
            raise
    
    def load_model(
        self,
        path: str,
        model: nn.Module | None = None,
        device: str = 'cpu'
    ) -> nn.Module | Dict[str, Any]:
        """Load model from artifact store.
        
        Args:
            path: Relative path within artifact store
            model: Optional model instance to load state into
            device: Device to load model onto ('cpu', 'cuda', etc.)
            
        Returns:
            Loaded model or state dict
        """
        full_path = self._normalize_path(path)
        
        try:
            # For local filesystem, use direct torch.load
            if self._get_protocol() == 'file':
                state_dict = torch.load(full_path, map_location=device)
            else:
                # For cloud storage, download to temporary location
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp_path = tmp.name
                
                try:
                    self.fs.download(full_path, tmp_path)
                    state_dict = torch.load(tmp_path, map_location=device)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            # If model instance provided, load state into it
            if model is not None:
                model.load_state_dict(state_dict)
                logger.info(f"Loaded model state from {full_path}")
                return model
            else:
                logger.info(f"Loaded state dict from {full_path}")
                return state_dict
                
        except Exception as e:
            logger.error(f"Failed to load model from {full_path}: {e}")
            raise
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        path: str
    ) -> str:
        """Save metadata JSON file.
        
        Args:
            metadata: Metadata dictionary
            path: Relative path within artifact store
            
        Returns:
            Full artifact URI of saved metadata
        """
        full_path = self._normalize_path(path)
        
        try:
            # Convert metadata to JSON-serializable format
            json_data = json.dumps(metadata, indent=2, default=str)
            
            if self._get_protocol() == 'file':
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(json_data)
            else:
                # For cloud storage
                with self.fs.open(full_path, 'w') as f:
                    f.write(json_data)
            
            logger.info(f"Saved metadata to {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to save metadata to {full_path}: {e}")
            raise
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load metadata JSON file.
        
        Args:
            path: Relative path within artifact store
            
        Returns:
            Loaded metadata dictionary
        """
        full_path = self._normalize_path(path)
        
        try:
            if self._get_protocol() == 'file':
                with open(full_path, 'r') as f:
                    return json.load(f)
            else:
                with self.fs.open(full_path, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to load metadata from {full_path}: {e}")
            raise
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str
    ) -> str:
        """Save a complete checkpoint (model, optimizer, metadata, etc.).
        
        Args:
            checkpoint: Checkpoint dictionary
            path: Relative path within artifact store
            
        Returns:
            Full artifact URI of saved checkpoint
        """
        full_path = self._normalize_path(path)
        
        try:
            if self._get_protocol() == 'file':
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                torch.save(checkpoint, full_path)
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
                    tmp_path = tmp.name
                    torch.save(checkpoint, tmp_path)
                
                try:
                    self.fs.upload(tmp_path, full_path)
                finally:
                    os.remove(tmp_path)
            
            logger.info(f"Saved checkpoint to {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {full_path}: {e}")
            raise
    
    def load_checkpoint(
        self,
        path: str,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """Load a complete checkpoint.
        
        Args:
            path: Relative path within artifact store
            device: Device to load checkpoint onto
            
        Returns:
            Loaded checkpoint dictionary
        """
        full_path = self._normalize_path(path)
        
        try:
            if self._get_protocol() == 'file':
                return torch.load(full_path, map_location=device)
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
                    tmp_path = tmp.name
                
                try:
                    self.fs.download(full_path, tmp_path)
                    return torch.load(tmp_path, map_location=device)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {full_path}: {e}")
            raise
    
    def list_artifacts(self, prefix: str = '') -> list[str]:
        """List artifacts in storage.
        
        Args:
            prefix: Optional path prefix to filter results
            
        Returns:
            List of artifact paths
        """
        search_path = self._normalize_path(prefix) if prefix else self.artifact_uri
        
        try:
            if self._get_protocol() == 'file':
                if not os.path.exists(search_path):
                    return []
                artifacts = []
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), self.artifact_uri)
                        artifacts.append(rel_path)
                return artifacts
            else:
                return self.fs.glob(f"{search_path}/**")
                
        except Exception as e:
            logger.error(f"Failed to list artifacts from {search_path}: {e}")
            return []
    
    def delete_artifact(self, path: str) -> bool:
        """Delete an artifact.
        
        Args:
            path: Relative path within artifact store
            
        Returns:
            True if successful, False otherwise
        """
        full_path = self._normalize_path(path)
        
        try:
            if self._get_protocol() == 'file':
                if os.path.exists(full_path):
                    os.remove(full_path)
                    logger.info(f"Deleted artifact {full_path}")
                    return True
            else:
                self.fs.rm(full_path)
                logger.info(f"Deleted artifact {full_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete artifact at {full_path}: {e}")
            return False
    
    def get_artifact_uri(self, path: str) -> str:
        """Get full artifact URI for a relative path.
        
        Args:
            path: Relative path within artifact store
            
        Returns:
            Full artifact URI
        """
        return self._normalize_path(path)


# Global artifact store instance
_artifact_store: ArtifactStore | None = None


def get_artifact_store(artifact_uri: str | None = None) -> ArtifactStore:
    """Get or create global artifact store instance.
    
    Args:
        artifact_uri: Optional artifact URI (only used on first call)
        
    Returns:
        Global ArtifactStore instance
    """
    global _artifact_store
    
    if _artifact_store is None:
        _artifact_store = ArtifactStore(artifact_uri)
    
    return _artifact_store


def set_artifact_store(artifact_uri: str) -> None:
    """Set global artifact store URI.
    
    Args:
        artifact_uri: New artifact storage URI
    """
    global _artifact_store
    _artifact_store = ArtifactStore(artifact_uri)
