"""Configuration management for benchmarking."""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    params: Dict[str, Any]
    task_type: str = "classification"  # classification, link_prediction, anomaly_detection, regression
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    num_nodes: int = 1000
    num_features: int = 16
    num_edges: int = 5000
    num_classes: int = 2
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    feature_noise: float = 0.1
    edge_noise: float = 0.1
    
    def __post_init__(self):
        """Validate ratios."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Data ratios must sum to 1.0, got {total}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""
    name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    description: str = ""
    output_dir: str = "./benchmark_results"
    artifact_uri: str = "./artifacts"  # Local path, s3://bucket/path, gs://bucket/path
    save_model: bool = True
    save_data: bool = False
    device: str = "auto"  # auto, cpu, cuda
    num_runs: int = 1
    verbose: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "output_dir": self.output_dir,
            "artifact_uri": self.artifact_uri,
            "save_model": self.save_model,
            "save_data": self.save_data,
            "device": self.device,
            "num_runs": self.num_runs,
            "verbose": self.verbose
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            model=ModelConfig.from_dict(data["model"]),
            data=DataConfig.from_dict(data["data"]),
            training=TrainingConfig.from_dict(data["training"]),
            output_dir=data.get("output_dir", "./benchmark_results"),
            artifact_uri=data.get("artifact_uri", "./artifacts"),
            save_model=data.get("save_model", True),
            save_data=data.get("save_data", False),
            device=data.get("device", "auto"),
            num_runs=data.get("num_runs", 1),
            verbose=data.get("verbose", True)
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BenchmarkConfig":
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ConfigManager:
    """Manages multiple benchmark configurations."""
    
    def __init__(self, config_dir: Union[str, Path] = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._configs: Dict[str, BenchmarkConfig] = {}
    
    def add_config(self, config: BenchmarkConfig) -> None:
        """Add a configuration."""
        self._configs[config.name] = config
        
        # Save to file
        filepath = self.config_dir / f"{config.name}.json"
        config.save(filepath)
    
    def get_config(self, name: str) -> BenchmarkConfig:
        """Get a configuration by name."""
        if name in self._configs:
            return self._configs[name]
        
        # Try to load from file
        filepath = self.config_dir / f"{name}.json"
        if filepath.exists():
            config = BenchmarkConfig.load(filepath)
            self._configs[name] = config
            return config
        
        raise KeyError(f"Configuration '{name}' not found")
    
    def list_configs(self) -> List[str]:
        """List all available configuration names."""
        # Load from files if not already loaded
        for filepath in self.config_dir.glob("*.json"):
            name = filepath.stem
            if name not in self._configs:
                try:
                    config = BenchmarkConfig.load(filepath)
                    self._configs[name] = config
                except Exception:
                    pass
        
        return list(self._configs.keys())
    
    def remove_config(self, name: str) -> None:
        """Remove a configuration."""
        if name in self._configs:
            del self._configs[name]
        
        # Remove file
        filepath = self.config_dir / f"{name}.json"
        if filepath.exists():
            filepath.unlink()
    
    def create_default_configs(self) -> None:
        """Create default benchmark configurations."""
        # GCN Classification
        gcn_config = BenchmarkConfig(
            name="gcn_classification",
            description="GCN model for node classification",
            model=ModelConfig(
                name="gcn",
                params={
                    "in_channels": 16,
                    "hidden_channels": 64,
                    "out_channels": 2,
                    "num_layers": 2,
                    "dropout": 0.5
                },
                task_type="classification"
            ),
            data=DataConfig(),
            training=TrainingConfig()
        )
        self.add_config(gcn_config)
        
        # Link Prediction
        link_config = BenchmarkConfig(
            name="link_prediction",
            description="Link prediction model",
            model=ModelConfig(
                name="link_predictor",
                params={
                    "in_channels": 16,
                    "hidden_channels": 64,
                    "num_layers": 2,
                    "dropout": 0.5
                },
                task_type="link_prediction"
            ),
            data=DataConfig(num_edges=10000),
            training=TrainingConfig(learning_rate=0.005)
        )
        self.add_config(link_config)
        
        # SAGE Encoder
        sage_config = BenchmarkConfig(
            name="sage_encoder",
            description="GraphSAGE encoder for inductive learning",
            model=ModelConfig(
                name="sage_encoder",
                params={
                    "in_channels": 16,
                    "hidden_channels": 64,
                    "out_channels": 64,
                    "num_layers": 2,
                    "dropout": 0.5
                },
                task_type="classification"
            ),
            data=DataConfig(num_nodes=2000),
            training=TrainingConfig()
        )
        self.add_config(sage_config)
        
        # Deep SVDD
        svdd_config = BenchmarkConfig(
            name="deep_svdd",
            description="Deep SVDD for anomaly detection",
            model=ModelConfig(
                name="deep_svdd",
                params={
                    "in_channels": 16,
                    "hidden_channels": 64,
                    "out_channels": 32,
                    "num_layers": 2,
                    "dropout": 0.5
                },
                task_type="anomaly_detection"
            ),
            data=DataConfig(num_classes=1),
            training=TrainingConfig(
                learning_rate=0.001,
                loss_function="deep_svdd"
            )
        )
        self.add_config(svdd_config)


def create_config_from_template(
    name: str,
    model_name: str,
    task_type: str = "classification",
    **kwargs
) -> BenchmarkConfig:
    """Create a configuration from a template."""
    
    # Default model parameters based on model type
    model_params = {
        "gcn": {
            "in_channels": 16,
            "hidden_channels": 64,
            "out_channels": 2,
            "num_layers": 2,
            "dropout": 0.5
        },
        "link_predictor": {
            "in_channels": 16,
            "hidden_channels": 64,
            "num_layers": 2,
            "dropout": 0.5
        },
        "sage_encoder": {
            "in_channels": 16,
            "hidden_channels": 64,
            "out_channels": 64,
            "num_layers": 2,
            "dropout": 0.5
        },
        "deep_svdd": {
            "in_channels": 16,
            "hidden_channels": 64,
            "out_channels": 32,
            "num_layers": 2,
            "dropout": 0.5
        }
    }
    
    # Update with provided kwargs
    if "model_params" in kwargs:
        model_params[model_name].update(kwargs["model_params"])
    
    # Adjust output channels based on task
    if task_type == "classification":
        model_params[model_name]["out_channels"] = kwargs.get("num_classes", 2)
    elif task_type == "anomaly_detection":
        model_params[model_name]["out_channels"] = kwargs.get("embedding_dim", 32)
    
    return BenchmarkConfig(
        name=name,
        description=kwargs.get("description", f"{model_name} model for {task_type}"),
        model=ModelConfig(
            name=model_name,
            params=model_params[model_name],
            task_type=task_type
        ),
        data=DataConfig(**kwargs.get("data_params", {})),
        training=TrainingConfig(**kwargs.get("training_params", {})),
        output_dir=kwargs.get("output_dir", "./benchmark_results"),
        device=kwargs.get("device", "auto"),
        num_runs=kwargs.get("num_runs", 1),
        verbose=kwargs.get("verbose", True)
    )


def validate_config(config: BenchmarkConfig) -> List[str]:
    """Validate a benchmark configuration and return any issues."""
    issues = []
    
    # Validate model
    valid_models = ["gcn", "link_predictor", "sage_encoder", "deep_svdd"]
    if config.model.name not in valid_models:
        issues.append(f"Invalid model name: {config.model.name}. Valid: {valid_models}")
    
    # Validate task type
    valid_tasks = ["classification", "link_prediction", "anomaly_detection", "regression"]
    if config.model.task_type not in valid_tasks:
        issues.append(f"Invalid task type: {config.model.task_type}. Valid: {valid_tasks}")
    
    # Validate data config
    if config.data.num_nodes <= 0:
        issues.append("Number of nodes must be positive")
    if config.data.num_features <= 0:
        issues.append("Number of features must be positive")
    if config.data.num_edges <= 0:
        issues.append("Number of edges must be positive")
    if config.data.num_classes <= 0:
        issues.append("Number of classes must be positive")
    
    # Validate training config
    if config.training.epochs <= 0:
        issues.append("Number of epochs must be positive")
    if config.training.batch_size <= 0:
        issues.append("Batch size must be positive")
    if config.training.learning_rate <= 0:
        issues.append("Learning rate must be positive")
    
    # Validate output directory
    try:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create output directory: {e}")
    
    return issues
