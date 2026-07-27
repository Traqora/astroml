"""Core benchmarking framework for GNN models on Stellar data."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import GCN, LinkPredictor, InductiveSAGEEncoder, DeepSVDD
from ..ingestion.service import IngestionService
from ..artifacts import get_artifact_store


@dataclass
class BenchmarkConfig:
    """Configuration for model benchmarking."""
    
    # Model configuration
    model_name: str
    model_params: Dict[str, Any]
    
    # Data configuration  
    start_ledger: int = 0
    end_ledger: int = 1000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Training configuration
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    early_stopping_patience: int = 10
    
    # Evaluation configuration
    metrics: List[str] = None
    device: str = "auto"
    random_seed: int = 42
    
    # Output configuration
    output_dir: str = "./benchmark_results"
    save_model: bool = True
    save_predictions: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")


@dataclass
class BenchmarkResult:
    """Results from a model benchmark run."""
    
    # Metadata
    model_name: str
    model_params: Dict[str, Any]
    timestamp: float
    device: str
    random_seed: int
    
    # Data statistics
    total_nodes: int
    total_edges: int
    train_nodes: int
    val_nodes: int
    test_nodes: int
    
    # Training metrics
    train_time: float
    epochs_trained: int
    best_epoch: int
    train_losses: List[float]
    val_losses: List[float]
    
    # Performance metrics
    metrics: Dict[str, float]
    
    # Resource usage
    peak_memory_mb: float
    gpu_memory_mb: Optional[float] = None
    
    # Additional info
    convergence_epoch: Optional[int] = None
    notes: Optional[str] = None


class ModelBenchmark:
    """Main benchmarking class for GNN models."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.training.seed)
                
            # Create output directory
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            
            self.model = None
            self.results = None
        
    def load_data(self) -> Dict[str, Any]:
        """Load and prepare Stellar data for benchmarking."""
        print(f"Loading synthetic data with {self.config.data.num_nodes} nodes")
        
        # For now, create synthetic data that mimics Stellar transaction graph
        num_nodes = self.config.data.num_nodes
        num_edges = self.config.data.num_edges
        
        # Create synthetic node features (transaction features)
        x = torch.randn(num_nodes, self.config.data.num_features)
        
        # Create synthetic edges (transactions between accounts)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Create synthetic labels (e.g., fraud detection)
        y = torch.randint(0, self.config.data.num_classes, (num_nodes,))
        
        # Split data
        train_mask, val_mask, test_mask = self._split_data(num_nodes)
        
        data = {
            'x': x,
            'edge_index': edge_index,
            'y': y,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }
        
        return data
    
    def _split_data(self, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split data into train/val/test sets."""
        indices = torch.randperm(num_nodes)
        
        train_size = int(self.config.data.train_ratio * num_nodes)
        val_size = int(self.config.data.val_ratio * num_nodes)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        return train_mask, val_mask, test_mask
    
    def create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create model based on configuration."""
        model_name = self.config.model.name.lower()
        params = self.config.model.params
        
        if model_name == "gcn":
            return GCN(
                input_dim=input_dim,
                hidden_dim=params.get("hidden_channels", 64),
                output_dim=output_dim,
                dropout=params.get("dropout", 0.5)
            )
        elif model_name == "link_predictor":
            return LinkPredictor(
                input_dim=input_dim,
                hidden_dims=[params.get("hidden_channels", 64)],
                embedding_dim=params.get("hidden_channels", 32),
                dropout=params.get("dropout", 0.5),
                decoder="dot"
            )
        elif model_name == "sage_encoder":
            return InductiveSAGEEncoder(
                input_dim=input_dim,
                hidden_dim=params.get("hidden_channels", 64),
                output_dim=output_dim,
                num_layers=params.get("num_layers", 2),
                dropout=params.get("dropout", 0.5)
            )
        elif model_name == "deep_svdd":
            return DeepSVDD(
                input_dim=input_dim,
                hidden_dim=params.get("hidden_channels", 64),
                embedding_dim=params.get("out_channels", 32)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_model(self, model: nn.Module, data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Train the model and return training history."""
        model = model.to(self.device)
        data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.training.epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            out = model(data['x'], data['edge_index'])
            train_loss = criterion(out[data['train_mask']], data['y'][data['train_mask']])
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data['x'], data['edge_index'])
                val_loss = criterion(val_out[data['val_mask']], data['y'][data['val_mask']])
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': len(train_losses) - patience_counter - 1
        }
    
    def evaluate_model(self, model: nn.Module, data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model and compute metrics."""
        model.eval()
        data = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in data.items()}
        
        with torch.no_grad():
            out = model(data['x'], data['edge_index'])
            pred = out.argmax(dim=1)
            y_true = data['y'][data['test_mask']]
            y_pred = pred[data['test_mask']]
            
            # Compute metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {}
            
            # Compute all standard metrics
            metrics["accuracy"] = accuracy_score(y_true.cpu(), y_pred.cpu())
            metrics["precision"] = precision_score(y_true.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
            metrics["recall"] = recall_score(y_true.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
            metrics["f1"] = f1_score(y_true.cpu(), y_pred.cpu(), average='weighted', zero_division=0)
            
            try:
                # Get probabilities for AUC
                probs = torch.softmax(out, dim=1)[:, 1][data['test_mask']]
                metrics["auc"] = roc_auc_score(y_true.cpu(), probs.cpu())
            except Exception:
                metrics["auc"] = 0.0
        
        return metrics
    
    def _measure_memory_usage(self) -> tuple[float, Optional[float]]:
        """Measure current memory usage."""
        import psutil
        import os
        
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # GPU memory if available
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            
        return cpu_memory, gpu_memory
    
    def run_benchmark(self) -> BenchmarkResult:
        """Run the complete benchmark."""
        print(f"Starting benchmark for {self.config.model.name}")
        
        # Load data
        data = self.load_data()
        input_dim = data['x'].shape[1]
        output_dim = len(torch.unique(data['y']))
        
        # Create model
        self.model = self.create_model(input_dim, output_dim)
        
        # Measure initial memory
        start_memory, start_gpu_memory = self._measure_memory_usage()
        
        # Train model
        start_time = time.time()
        training_history = self.train_model(self.model, data)
        train_time = time.time() - start_time
        
        # Measure peak memory
        peak_memory, peak_gpu_memory = self._measure_memory_usage()
        
        # Evaluate model
        metrics = self.evaluate_model(self.model, data)
        
        # Create result object
        result = BenchmarkResult(
            model_name=self.config.model.name,
            model_params=self.config.model.params,
            timestamp=time.time(),
            device=str(self.device),
            random_seed=self.config.training.seed,
            total_nodes=data['x'].shape[0],
            total_edges=data['edge_index'].shape[1],
            train_nodes=data['train_mask'].sum().item(),
            val_nodes=data['val_mask'].sum().item(),
            test_nodes=data['test_mask'].sum().item(),
            train_time=train_time,
            epochs_trained=len(training_history['train_losses']),
            best_epoch=training_history['best_epoch'],
            train_losses=training_history['train_losses'],
            val_losses=training_history['val_losses'],
            metrics=metrics,
            peak_memory_mb=peak_memory,
            gpu_memory_mb=peak_gpu_memory,
            convergence_epoch=training_history['best_epoch']
        )
        
        self.results = result
        
        # Save results
        self._save_results(result)
        
        # Save configuration with environment info for reproducibility
        self._save_config()
        
        if self.config.save_model:
            self._save_model()
            
        print(f"Benchmark completed in {train_time:.2f}s")
        print(f"Metrics: {metrics}")
        
        return result
    
    def _save_results(self, result: BenchmarkResult):
        """Save benchmark results and configuration to file for reproducibility.
        
        Saves:
        - result.json: Benchmark results with all metrics
        - config.json: Full configuration including random seed
        - metadata.json: Metadata linking config and result
        """
        import time
        from datetime import datetime
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique run identification
        timestamp = datetime.utcnow().isoformat()
        run_id = f"{result.model_name}_{int(result.timestamp)}"
        
        # Save result
        result_dict = asdict(result)
        result_path = output_dir / f"{run_id}_result.json"
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"Results saved to {result_path}")
        
        # Save configuration for reproducibility
        config_dict = asdict(self.config)
        config_path = output_dir / f"{run_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Configuration saved to {config_path}")
        
        # Save metadata linking config and result
        metadata = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_name": result.model_name,
            "random_seed": result.random_seed,
            "device": result.device,
            "config_file": str(config_path),
            "result_file": str(result_path),
            "train_time_seconds": result.train_time,
            "epochs_trained": result.epochs_trained,
            "best_metrics": result.metrics,
        }
        metadata_path = output_dir / f"{run_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
    
    def _save_config(self):
        """Save benchmark configuration with environment info for reproducibility."""
        from .utils import get_environment_info
        
        config_path = Path(self.config.output_dir) / "benchmark-config.yaml"
        
        # Create output directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect environment information
        env_info = get_environment_info()
        
        # Build config dict with environment info
        config_dict = {
            'benchmark_config': self.config.to_dict(),
            'environment': env_info,
            'timestamp': time.time()
        }
        
        # Save as YAML
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to {config_path}")
    
    def _save_model(self):
        """Save trained model to artifact store."""
        if self.model is not None:
            # Initialize artifact store with configured URI
            store = get_artifact_store(self.config.artifact_uri)
            
            # Create model filename
            model_filename = f"{self.config.model.name}_model.pt"
            
            # Save model to artifact store
            try:
                artifact_uri = store.save_model(
                    self.model,
                    model_filename,
                    metadata={
                        'model_name': self.config.model.name,
                        'model_params': self.config.model.params,
                        'timestamp': time.time(),
                    }
                )
                print(f"Model saved to artifact store: {artifact_uri}")
            except Exception as e:
                print(f"Warning: Failed to save model to artifact store: {e}")
                # Fallback to local save
                model_path = Path(self.config.output_dir) / model_filename
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved locally to {model_path}")
