"""MLflow experiment tracking integration for AstroML."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Thin MLflow wrapper used by training scripts.

    Gracefully degrades to a no-op when MLflow is not installed or
    when ``enabled=False`` so training still works without the dependency.
    """

    def __init__(
        self,
        enabled: bool = True,
        tracking_uri: str = "mlruns",
        experiment_name: str = "astroml_experiment",
        run_name: Optional[str] = None,
        log_model_weights: bool = True,
    ):
        self.enabled = enabled
        self.log_model_weights = log_model_weights
        self._run = None

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
    ) -> None:
        """Log model weights as an MLflow artifact.

        Saves ``model.state_dict()`` to a temporary ``.pth`` file and
        uploads it.  If *checkpoint_path* already exists on disk it is
        uploaded directly (avoids a redundant save).
        """
        if not self.enabled or self._run is None or not self.log_model_weights:
            return

        import tempfile, os

        if checkpoint_path and Path(checkpoint_path).exists():
            self._mlflow.log_artifact(checkpoint_path, artifact_path=artifact_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                torch.save(model.state_dict(), tmp.name)
                self._mlflow.log_artifact(tmp.name, artifact_path=artifact_path)
                os.unlink(tmp.name)

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
