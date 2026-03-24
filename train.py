#!/usr/bin/env python3
"""
Training script for AstroML experiments using Hydra configuration.

Usage:
    python train.py                    # Use default config
    python train.py model.lr=0.001     # Override learning rate
    python train.py experiment=debug   # Use debug experiment
    python train.py --multirun model.lr=0.001,0.01,0.1  # Hyperparameter sweep
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, get_original_cwd

from astroml.models.gcn import GCN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_device(device_config: str) -> torch.device:
    """Set up the computation device based on configuration."""
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    logger.info(f"Using device: {device}")
    return device


def load_dataset(cfg: DictConfig) -> Any:
    """Load and prepare the dataset."""
    logger.info(f"Loading dataset: {cfg.data.name}")
    
    # Instantiate dataset from config
    dataset = instantiate(cfg.data)
    data = dataset[0]
    
    logger.info(f"Dataset loaded: {dataset.data}")
    logger.info(f"Number of classes: {dataset.num_classes}")
    logger.info(f"Number of node features: {dataset.num_node_features}")
    
    return dataset, data


def create_model(cfg: DictConfig, dataset: Any) -> torch.nn.Module:
    """Create and initialize the model."""
    # Update model dimensions based on dataset
    model_cfg = cfg.model.copy()
    model_cfg.input_dim = dataset.num_node_features
    model_cfg.output_dim = dataset.num_classes
    
    logger.info(f"Creating model with config: {model_cfg}")
    model = instantiate(model_cfg)
    
    return model


def create_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    optimizer_cfg = {
        "params": model.parameters(),
        "lr": cfg.training.lr,
    }
    
    # Add optimizer-specific parameters
    if cfg.training.optimizer == "adam":
        optimizer_cfg.update(cfg.training.optimizer_configs.adam)
    elif cfg.training.optimizer == "sgd":
        optimizer_cfg.update(cfg.training.optimizer_configs.sgd)
    elif cfg.training.optimizer == "adamw":
        optimizer_cfg.update(cfg.training.optimizer_configs.adamw)
    
    logger.info(f"Creating {cfg.training.optimizer} optimizer with lr={cfg.training.lr}")
    optimizer = getattr(torch.optim, cfg.training.optimizer.upper())(**optimizer_cfg)
    
    return optimizer


def train_epoch(model: torch.nn.Module, data: Any, optimizer: torch.optim.Optimizer, 
                device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.nll_loss(out[data.train_mask], data.y.to(device)[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model: torch.nn.Module, data: Any, device: torch.device, 
             mask_name: str = "test_mask") -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)
        
        mask = getattr(data, mask_name)
        correct = (pred[mask] == data.y.to(device)[mask]).sum()
        accuracy = int(correct) / int(mask.sum())
        
        # Calculate loss
        loss = F.nll_loss(out[mask], data.y.to(device)[mask]).item()
    
    return {"accuracy": accuracy, "loss": loss}


def train(cfg: DictConfig) -> Dict[str, Any]:
    """Main training function."""
    # Set up device
    device = set_device(cfg.experiment.device)
    
    # Load dataset
    dataset, data = load_dataset(cfg)
    data = data.to(device)
    
    # Create model
    model = create_model(cfg, dataset)
    model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(cfg, model)
    
    # Training loop
    logger.info(f"Starting training for {cfg.training.epochs} epochs")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(cfg.training.epochs):
        # Train
        train_loss = train_epoch(model, data, optimizer, device)
        
        # Evaluate
        train_metrics = evaluate(model, data, device, "train_mask")
        val_metrics = evaluate(model, data, device, "val_mask")
        
        # Log progress
        if epoch % cfg.training.log_interval == 0:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # Save best model
            if cfg.training.save_best_only:
                torch.save(model.state_dict(), 
                          Path(cfg.experiment.save_dir) / "best_model.pth")
        else:
            patience_counter += 1
        
        if (cfg.training.early_stopping.patience > 0 and 
            patience_counter >= cfg.training.early_stopping.patience):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    test_metrics = evaluate(model, data, device, "test_mask")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save final model
    if cfg.training.save_last:
        torch.save(model.state_dict(), 
                  Path(cfg.experiment.save_dir) / "last_model.pth")
    
    # Save configuration
    OmegaConf.save(cfg, Path(cfg.experiment.save_dir) / "config.yaml")
    
    return {
        "test_accuracy": test_metrics['accuracy'],
        "test_loss": test_metrics['loss'],
        "best_val_accuracy": best_val_acc,
        "epochs_trained": epoch + 1
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Create save directory
    save_dir = Path(cfg.experiment.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    if cfg.experiment.seed is not None:
        torch.manual_seed(cfg.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.experiment.seed)
    
    # Run training
    results = train(cfg)
    
    # Log results
    logger.info("Training completed!")
    logger.info(f"Results: {results}")
    
    # Save results
    results_path = save_dir / "results.yaml"
    OmegaConf.save(OmegaConf.create(results), results_path)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
