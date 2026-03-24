#!/usr/bin/env python3
"""
Test script for Hydra configuration system (without PyTorch dependencies).

Usage:
    python test_config.py                    # Use default config
    python test_config.py model.lr=0.001     # Override learning rate
    python test_config.py experiment=debug   # Use debug experiment
"""

import logging
from pathlib import Path
from typing import Dict, Any

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config(cfg: DictConfig) -> Dict[str, Any]:
    """Test function that just prints the configuration."""
    logger.info("Configuration test successful!")
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Test accessing configuration values
    results = {
        "experiment_name": cfg.experiments.experiment.name,
        "model_hidden_dims": cfg.experiments.model.hidden_dims,
        "training_lr": cfg.experiments.training.lr,
        "training_epochs": cfg.experiments.training.epochs,
        "data_name": cfg.experiments.data.name,
    }
    
    logger.info(f"Test results: {results}")
    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Create save directory
    save_dir = Path(cfg.experiments.experiment.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Run test
    results = test_config(cfg)
    
    # Log results
    logger.info("Configuration test completed!")
    logger.info(f"Results: {results}")
    
    # Save configuration
    OmegaConf.save(cfg, save_dir / "config.yaml")
    OmegaConf.save(OmegaConf.create(results), save_dir / "results.yaml")
    
    logger.info(f"Configuration saved to {save_dir}")


if __name__ == "__main__":
    main()
