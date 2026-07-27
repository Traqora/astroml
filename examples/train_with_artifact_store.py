"""Example training script using configurable artifact storage.

This example demonstrates how to use the artifact storage system to save
models to local filesystem, S3, or GCS.

Usage:
    # Local storage (default)
    python examples/train_with_artifact_store.py

    # S3 storage
    python examples/train_with_artifact_store.py artifact_storage=s3

    # GCS storage
    python examples/train_with_artifact_store.py artifact_storage=gcs
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from astroml.storage import create_artifact_store
from astroml.tracking import MLflowTracker

logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_example():
    """Example training with artifact storage."""
    # Initialize Hydra config
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config")

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create artifact store from config
    artifact_uri = cfg.training.artifact_storage.get_artifact_uri()
    logger.info(f"Using artifact store: {artifact_uri}")

    artifact_store = create_artifact_store(artifact_uri)

    # Initialize MLflow tracker with artifact store
    tracker = MLflowTracker(
        enabled=cfg.mlflow.enabled,
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        artifact_store=artifact_store,
    )

    # Create model
    model = SimpleModel(input_dim=10, hidden_dim=64, output_dim=2)
    logger.info(f"Model created: {model}")

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    tracker.log_params({"total_parameters": total_params})

    # Simulate training
    logger.info("Starting training simulation...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        # Simulate batch
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        tracker.log_metric("loss", loss.item(), step=epoch)
        logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")

    # Save model checkpoint
    checkpoint_path = Path("best_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model checkpoint saved to {checkpoint_path}")

    # Log model artifact to both MLflow and artifact store
    artifact_uri = tracker.log_model_artifact(
        model=model,
        artifact_path="model",
        checkpoint_path=str(checkpoint_path),
    )
    logger.info(f"Model artifact saved to: {artifact_uri}")

    # Save training config as artifact
    config_path = Path("training_config.yaml")
    OmegaConf.save(cfg, config_path)
    config_uri = tracker.save_artifact(config_path, artifact_path="config")
    logger.info(f"Config artifact saved to: {config_uri}")

    # List all artifacts in store
    logger.info("Artifacts in store:")
    artifacts = artifact_store.list_artifacts()
    for artifact in artifacts:
        logger.info(f"  - {artifact}")

    # Cleanup
    checkpoint_path.unlink()
    config_path.unlink()

    tracker.end()
    logger.info("Training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_example()
