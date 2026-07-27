"""Celery task: train an AstroML model — issue #296.

In production this task would load a dataset from the feature store, set up a
PyTorch Lightning trainer, and persist model weights and MLflow metrics.
The current implementation is a placeholder that validates the config dict
and returns a training summary without touching the GPU.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from astroml.tasks.celery_app import app

logger = logging.getLogger(__name__)


@app.task(
    name="astroml.tasks.model_train_task.train_model",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 5},
)
def train_model(self, model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Train a named model with the given configuration.

    Parameters
    ----------
    model_name:
        Identifier of the model architecture to train (e.g. ``link_predictor``).
    config:
        Hyper-parameter dict. Recognised keys (all optional):
            epochs      — number of training epochs (default 10)
            lr          — learning rate (default 1e-3)
            hidden_dim  — GNN hidden dimension (default 64)

    Returns
    -------
    dict with keys:
        model_name  — echoed input
        status      — ``"trained"``
        epochs      — number of epochs trained
    """
    epochs = int(config.get("epochs", 10))

    logger.info("train_model started: model=%s epochs=%d", model_name, epochs)

    # Simulate training time proportional to epochs.
    time.sleep(0.01 * epochs)

    result = {
        "model_name": model_name,
        "status": "trained",
        "epochs": epochs,
    }
    logger.info("train_model finished: %s", result)
    return result
