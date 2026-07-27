"""MLflow experiment tracking integration for AstroML."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from astroml.storage import ArtifactStore, create_artifact_store
from astroml.db.session import get_session
from api.models.orm import ModelRegistry

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Thin MLflow wrapper used by training scripts.

    Gracefully degrades to a no-op when MLflow is not installed or
    when ``enabled=False`` so training still works without the dependency.
    
    Supports configurable artifact storage backends (local, S3, GCS) via
    the artifact_uri parameter.
    """

    def __init__(
        self,
        enabled: bool = True,
        tracking_uri: str = "mlruns",
        experiment_name: str = "astroml_experiment",
        run_name: Optional[str] = None,
        log_model_weights: bool = True,
        artifact_uri: Optional[str] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ):
        """Initialize MLflow tracker with optional artifact store.
        
        Args:
            enabled: Whether to enable MLflow tracking
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            run_name: Name of the run (auto-generated if None)
            log_model_weights: Whether to log model weights
            artifact_uri: URI for artifact storage (e.g., "file:///path", "s3://bucket/prefix")
            artifact_store: Pre-configured ArtifactStore instance (takes precedence over artifact_uri)
        """
        self.enabled = enabled
        self.log_model_weights = log_model_weights
        self._run = None
        self.artifact_store = artifact_store

        if not self.enabled:
            return

        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._run = mlflow.start_run(run_name=run_name)
            logger.info(
                "MLflow run started | experiment=%s run_id=%s",
                experiment_name,
                self._run.info.run_id,
            )
        except ImportError:
            logger.warning(
                "mlflow package not found — tracking disabled. "
                "Install it with: pip install mlflow"
            )
            self.enabled = False
            return

        # Initialize artifact store if provided
        if artifact_uri and not artifact_store:
            try:
                self.artifact_store = create_artifact_store(artifact_uri)
                logger.info(f"Artifact store initialized: {artifact_uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize artifact store: {e}")
                self.artifact_store = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a flat dictionary of hyper-parameters."""
        if not self.enabled or self._run is None:
            return
        self._mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar metric."""
        if not self.enabled or self._run is None:
            return
        self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar metrics at once."""
        if not self.enabled or self._run is None:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_model_artifact(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        checkpoint_path: Optional[str] = None,
    ) -> Optional[str]:
        """Log model weights as an MLflow artifact.

        Saves ``model.state_dict()`` to a temporary ``.pth`` file and
        uploads it. If *checkpoint_path* already exists on disk it is
        uploaded directly (avoids a redundant save).
        
        If an artifact store is configured, also saves to the artifact store
        and returns the artifact URI.

        Args:
            model: PyTorch model to log
            artifact_path: Path within MLflow artifacts
            checkpoint_path: Optional existing checkpoint file to log

        Returns:
            Artifact URI if artifact store is configured, None otherwise
        """
        if not self.enabled or self._run is None or not self.log_model_weights:
            return None

        import os

        artifact_uri = None
        
        # Determine which file to log
        if checkpoint_path and Path(checkpoint_path).exists():
            file_to_log = checkpoint_path
            should_cleanup = False
        else:
            # Create temporary file
            tmp_file = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
            tmp_file.close()
            torch.save(model.state_dict(), tmp_file.name)
            file_to_log = tmp_file.name
            should_cleanup = True

        try:
            # Log to MLflow
            self._mlflow.log_artifact(file_to_log, artifact_path=artifact_path)
            
            # Log to artifact store if configured
            if self.artifact_store:
                remote_path = f"{artifact_path}/{Path(file_to_log).name}"
                artifact_uri = self.artifact_store.save(file_to_log, remote_path)
                logger.info(f"Model artifact saved to store: {artifact_uri}")
        finally:
            # Cleanup temporary file if created
            if should_cleanup and Path(file_to_log).exists():
                os.unlink(file_to_log)

        return artifact_uri

    def save_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: str = "artifacts",
    ) -> Optional[str]:
        """Save an arbitrary artifact to both MLflow and artifact store.

        Args:
            local_path: Path to local file to save
            artifact_path: Path within artifact storage

        Returns:
            Artifact URI if artifact store is configured, None otherwise
        """
        if not self.enabled or self._run is None:
            return None

        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Artifact not found: {local_path}")

        # Log to MLflow
        self._mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

        # Log to artifact store if configured
        artifact_uri = None
        if self.artifact_store:
            remote_path = f"{artifact_path}/{local_path.name}"
            artifact_uri = self.artifact_store.save(local_path, remote_path)
            logger.info(f"Artifact saved to store: {artifact_uri}")

        return artifact_uri

    def load_artifact(
        self,
        remote_path: str,
        local_path: Union[str, Path],
    ) -> Path:
        """Load an artifact from the artifact store to local filesystem.

        Args:
            remote_path: Path in artifact store
            local_path: Destination path on local filesystem

        Returns:
            Path to loaded file

        Raises:
            RuntimeError: If no artifact store is configured
        """
        if not self.artifact_store:
            raise RuntimeError("No artifact store configured")

        return self.artifact_store.load(remote_path, local_path)

    def log_roc_auc(self, y_true: np.ndarray, y_score: np.ndarray, step: Optional[int] = None) -> None:
        """Compute and log ROC-AUC."""
        if not self.enabled or self._run is None:
            return
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_true, y_score)
            self.log_metric("roc_auc", auc, step=step)
        except Exception as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)

    def register_model(
        self,
        model_name: str,
        version: str,
        path: str,
        owner: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: Optional[str] = "inactive",
    ) -> Optional[ModelRegistry]:
        """Register a model version to the model registry.
        
        Automatically syncs metrics and parameters from the current MLflow run
        and stores the mlflow_run_id.
        
        Args:
            model_name: Name of the model
            version: Version identifier
            path: Path to model file
            owner: Optional owner of the model
            tags: Optional tags for the model
            status: Optional status (inactive, active, deprecated)
            
        Returns:
            The created ModelRegistry entry if successful, None otherwise
        """
        if not self.enabled or self._run is None:
            logger.warning("MLflow not enabled, skipping model registration")
            return None

        db = get_session()
        try:
            # Sync metrics from MLflow run
            metrics = {}
            if self._run.data.metrics:
                metrics = dict(self._run.data.metrics)

            # Check if model already exists
            existing = db.query(ModelRegistry).filter(
                ModelRegistry.name == model_name,
                ModelRegistry.version == version
            ).first()
            if existing:
                logger.warning(
                    f"Model {model_name} version {version} already exists, "
                    f"updating mlflow_run_id and metrics"
                )
                existing.mlflow_run_id = self._run.info.run_id
                existing.metrics = metrics
                db.commit()
                db.refresh(existing)
                return existing

            # Create new model entry
            entry = ModelRegistry(
                name=model_name,
                version=version,
                path=path,
                owner=owner,
                tags=tags,
                mlflow_run_id=self._run.info.run_id,
                metrics=metrics,
                status=status or "inactive",
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            logger.info(
                "Model registered | name=%s version=%s mlflow_run_id=%s",
                model_name,
                version,
                self._run.info.run_id
            )
            return entry
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            db.rollback()
            return None
        finally:
            db.close()

    def load_run_metadata(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Load MLflow run metadata for a registered model version.
        
        Args:
            model_name: Name of the model
            version: Version identifier
            
        Returns:
            Run metadata dict if found, None otherwise
        """
        if not self.enabled:
            logger.warning("MLflow not enabled, cannot load run metadata")
            return None

        db = get_session()
        try:
            entry = db.query(ModelRegistry).filter(
                ModelRegistry.name == model_name,
                ModelRegistry.version == version
            ).first()
            if not entry or not entry.mlflow_run_id:
                logger.warning(f"No MLflow run ID found for {model_name} v{version}")
                return None

            # Get run data from MLflow
            run = self._mlflow.get_run(entry.mlflow_run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "artifact_uri": run.info.artifact_uri,
            }
        except Exception as e:
            logger.error(f"Failed to load run metadata: {e}")
            return None
        finally:
            db.close()

    def end(self) -> None:
        """End the active MLflow run."""
        if self.enabled and self._run is not None:
            self._mlflow.end_run()
            logger.info("MLflow run ended.")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MLflowTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.end()
