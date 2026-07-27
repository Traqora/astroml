import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

from ..models.temporal import (
    TemporalGCN, TemporalGraphSAGE, TemporalGAT, 
    TemporalGraphTransformer, TemporalModelFactory
)
from ..utils.temporal import TemporalMetrics, TemporalAugmentation


@dataclass
class TemporalTrainingConfig:
    """Configuration for temporal model training."""
    model_type: str = "temporal_gcn"
    input_dim: int = 64
    hidden_dims: List[int] = None
    output_dim: int = 2
    temporal_dim: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    dropout: float = 0.5
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    temporal_consistency_weight: float = 0.1
    augmentation_enabled: bool = True
    augmentation_prob: float = 0.3
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class TemporalDataset(Dataset):
    """Dataset for temporal graph data."""
    
    def __init__(
        self,
        graphs: List[Dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
        augmentation: Optional[TemporalAugmentation] = None,
        augmentation_prob: float = 0.3
    ):
        self.graphs = graphs
        self.labels = labels
        self.augmentation = augmentation
        self.augmentation_prob = augmentation_prob
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        graph_data = self.graphs[idx].copy()
        
        # Apply augmentation if enabled
        if self.augmentation and np.random.random() < self.augmentation_prob:
            aug_type = np.random.choice(['noise', 'dropout', 'scaling'])
            
            if aug_type == 'noise':
                graph_data = self.augmentation.temporal_noise_augmentation(graph_data)
            elif aug_type == 'dropout':
                graph_data = self.augmentation.temporal_dropout(graph_data)
            elif aug_type == 'scaling':
                graph_data = self.augmentation.temporal_scaling(graph_data)
        
        # Add labels if available
        if self.labels is not None:
            graph_data['labels'] = self.labels[idx]
        
        return graph_data


class TemporalTrainer:
    """Trainer for temporal GNN models."""
    
    def __init__(self, config: TemporalTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.temporal_metrics = TemporalMetrics()
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'temporal_auc': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        if self.config.model_type == "temporal_gcn":
            return TemporalModelFactory.create_temporal_gcn(self.config)
        elif self.config.model_type == "temporal_sage":
            return TemporalModelFactory.create_temporal_sage(self.config)
        elif self.config.model_type == "temporal_gat":
            return TemporalModelFactory.create_temporal_gat(self.config)
        elif self.config.model_type == "temporal_transformer":
            return TemporalModelFactory.create_temporal_transformer(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self._forward_pass(batch)
            labels = batch['labels']
            
            # Compute loss
            classification_loss = self.criterion(predictions, labels)
            
            # Add temporal consistency loss
            temporal_loss = self.temporal_metrics.temporal_consistency_loss(
                predictions, batch['node_times'], self.config.temporal_consistency_weight
            )
            
            total_loss_batch = classification_loss + temporal_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            pred_classes = torch.argmax(predictions, dim=1)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_timestamps = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self._forward_pass(batch)
                labels = batch['labels']
                
                # Compute loss
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()
                
                # Update metrics
                pred_classes = torch.argmax(predictions, dim=1)
                correct += (pred_classes == labels).sum().item()
                total += labels.size(0)
                
                # Collect for temporal AUC
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_timestamps.append(batch['node_times'].cpu())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        # Compute temporal AUC
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_timestamps = torch.cat(all_timestamps, dim=0)
        
        temporal_auc = self.temporal_metrics.temporal_auc(
            all_predictions, all_labels, all_timestamps
        )
        
        return avg_loss, accuracy, temporal_auc
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the model."""
        if self.config.model_type == "temporal_transformer":
            return self.model(
                x=batch['node_features'],
                edge_index=batch['edge_index'],
                node_time=batch.get('node_times'),
                edge_time=batch.get('edge_times')
            )
        else:
            return self.model(
                x=batch['node_features'],
                edge_index=batch['edge_index'],
                edge_time=batch.get('edge_times'),
                node_time=batch.get('node_times')
            )
    
    def train(
        self,
        train_graphs: List[Dict[str, torch.Tensor]],
        train_labels: torch.Tensor,
        val_graphs: Optional[List[Dict[str, torch.Tensor]]] = None,
        val_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """Train the temporal model."""
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        
        # Create datasets
        augmentation = TemporalAugmentation() if self.config.augmentation_enabled else None
        
        train_dataset = TemporalDataset(
            train_graphs, train_labels, augmentation, self.config.augmentation_prob
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        # Validation setup
        val_loader = None
        if val_graphs is not None and val_labels is not None:
            val_dataset = TemporalDataset(val_graphs, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate epoch
            if val_loader is not None:
                val_loss, val_acc, temporal_auc = self.validate_epoch(val_loader)
                self.scheduler.step(val_loss)
            else:
                val_loss, val_acc, temporal_auc = train_loss, train_acc, 0.0
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['temporal_auc'].append(temporal_auc)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Temporal AUC: {temporal_auc:.4f}"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.training_history
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, f'temporal_model_checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint.
        
        Returns:
            True if checkpoint was loaded successfully, False otherwise.
        
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            ValueError: If checkpoint is corrupted or missing required keys
            RuntimeError: If state dict does not match model architecture
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}")
        
        # Validate required keys
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'training_history']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing required key: {key}")
        
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Model state dict does not match architecture: {e}")
        
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Optimizer state dict does not match: {e}")
        
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Scheduler state dict does not match: {e}")
        
        self.training_history = checkpoint['training_history']
        
        if 'epoch' in checkpoint:
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            self.logger.info("Loaded checkpoint (epoch info not available)")
        
        return True
    
    def evaluate(
        self,
        test_graphs: List[Dict[str, torch.Tensor]],
        test_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate the model on test data."""
        test_dataset = TemporalDataset(test_graphs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        test_loss, test_acc, temporal_auc = self.validate_epoch(test_loader)
        
        # Additional temporal metrics
        all_predictions = []
        all_labels = []
        all_timestamps = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                predictions = self._forward_pass(batch)
                labels = batch['labels']
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_timestamps.append(batch['node_times'].cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_timestamps = torch.cat(all_timestamps, dim=0)
        
        # Temporal prediction accuracy
        temporal_acc = self.temporal_metrics.temporal_prediction_accuracy(
            all_predictions, all_labels, all_timestamps
        )
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'temporal_auc': temporal_auc,
            'temporal_accuracy': temporal_acc
        }


class TemporalHyperparameterSearch:
    """Hyperparameter search for temporal models."""
    
    def __init__(self, search_space: Dict[str, List[Any]]):
        self.search_space = search_space
        self.results = []
    
    def random_search(
        self,
        train_graphs: List[Dict[str, torch.Tensor]],
        train_labels: torch.Tensor,
        val_graphs: List[Dict[str, torch.Tensor]],
        val_labels: torch.Tensor,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """Perform random hyperparameter search."""
        best_score = 0.0
        best_params = None
        
        for trial in range(n_trials):
            # Sample hyperparameters
            params = self._sample_params()
            
            # Create config
            config = TemporalTrainingConfig(**params)
            
            # Train model
            trainer = TemporalTrainer(config)
            history = trainer.train(train_graphs, train_labels, val_graphs, val_labels)
            
            # Evaluate
            val_acc = max(history['val_acc'])
            
            # Update best
            if val_acc > best_score:
                best_score = val_acc
                best_params = params
            
            # Store results
            self.results.append({
                'trial': trial,
                'params': params,
                'val_accuracy': val_acc,
                'training_history': history
            })
            
            print(f"Trial {trial+1}/{n_trials}: Val Acc = {val_acc:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }
    
    def _sample_params(self) -> Dict[str, Any]:
        """Sample random hyperparameters."""
        params = {}
        
        for key, values in self.search_space.items():
            if isinstance(values, list):
                params[key] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                if isinstance(values[0], int):
                    params[key] = np.random.randint(values[0], values[1] + 1)
                else:
                    params[key] = np.random.uniform(values[0], values[1])
        
        return params


class TemporalExperiment:
    """Experiment management for temporal models."""
    
    def __init__(self, name: str, config: TemporalTrainingConfig):
        self.name = name
        self.config = config
        self.timestamp = datetime.now()
        self.results = {}
    
    def run_experiment(
        self,
        train_graphs: List[Dict[str, torch.Tensor]],
        train_labels: torch.Tensor,
        val_graphs: List[Dict[str, torch.Tensor]],
        val_labels: torch.Tensor,
        test_graphs: List[Dict[str, torch.Tensor]],
        test_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Run complete experiment."""
        print(f"Running experiment: {self.name}")
        print(f"Model: {self.config.model_type}")
        print(f"Temporal dim: {self.config.temporal_dim}")
        
        # Train model
        trainer = TemporalTrainer(self.config)
        training_history = trainer.train(train_graphs, train_labels, val_graphs, val_labels)
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_graphs, test_labels)
        
        # Store results
        self.results = {
            'experiment_name': self.name,
            'config': self.config,
            'timestamp': self.timestamp,
            'training_history': training_history,
            'test_results': test_results
        }
        
        # Print summary
        print("\nExperiment Results:")
        print(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"Temporal AUC: {test_results['temporal_auc']:.4f}")
        print(f"Temporal Accuracy: {test_results['temporal_accuracy']:.4f}")
        
        return self.results
    
    def save_results(self, filepath: str):
        """Save experiment results."""
        import json
        
        # Convert to serializable format
        serializable_results = {
            'experiment_name': self.results['experiment_name'],
            'config': self.results['config'].__dict__,
            'timestamp': self.results['timestamp'].isoformat(),
            'training_history': {
                k: v for k, v in self.results['training_history'].items()
            },
            'test_results': self.results['test_results']
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")


# Default search space for hyperparameter optimization
DEFAULT_SEARCH_SPACE = {
    'model_type': ['temporal_gcn', 'temporal_sage', 'temporal_gat'],
    'hidden_dims': [[64, 32], [128, 64], [256, 128]],
    'temporal_dim': [16, 32, 64],
    'learning_rate': (0.0001, 0.01),
    'dropout': (0.1, 0.5),
    'temporal_consistency_weight': (0.01, 0.5),
    'augmentation_prob': (0.1, 0.5)
}


def create_temporal_experiment_suite(
    train_graphs: List[Dict[str, torch.Tensor]],
    train_labels: torch.Tensor,
    val_graphs: List[Dict[str, torch.Tensor]],
    val_labels: torch.Tensor,
    test_graphs: List[Dict[str, torch.Tensor]],
    test_labels: torch.Tensor
) -> List[Dict[str, Any]]:
    """Create a suite of temporal experiments."""
    
    experiments = []
    
    # Experiment configurations
    configs = [
        # Temporal GCN experiments
        TemporalTrainingConfig(
            model_type="temporal_gcn",
            hidden_dims=[128, 64],
            temporal_dim=32,
            temporal_consistency_weight=0.1
        ),
        TemporalTrainingConfig(
            model_type="temporal_gcn",
            hidden_dims=[256, 128],
            temporal_dim=64,
            temporal_consistency_weight=0.2
        ),
        
        # Temporal GraphSAGE experiments
        TemporalTrainingConfig(
            model_type="temporal_sage",
            hidden_dims=[128, 64],
            temporal_dim=32,
            temporal_consistency_weight=0.1
        ),
        
        # Temporal GAT experiments
        TemporalTrainingConfig(
            model_type="temporal_gat",
            hidden_dims=[128, 64],
            temporal_dim=32,
            temporal_consistency_weight=0.1
        ),
        
        # Temporal Transformer experiments
        TemporalTrainingConfig(
            model_type="temporal_transformer",
            hidden_dim=128,
            temporal_dim=64,
            temporal_consistency_weight=0.1
        ),
    ]
    
    # Run experiments
    for i, config in enumerate(configs):
        experiment = TemporalExperiment(f"temporal_exp_{i+1}_{config.model_type}", config)
        results = experiment.run_experiment(
            train_graphs, train_labels,
            val_graphs, val_labels,
            test_graphs, test_labels
        )
        
        experiments.append(results)
        
        # Save results
        experiment.save_results(f"temporal_experiment_{i+1}_results.json")
    
    # Compare results
    print("\nExperiment Comparison:")
    print("-" * 80)
    for i, result in enumerate(experiments):
        config = result['config']
        test_results = result['test_results']
        
        print(f"Experiment {i+1} ({config.model_type}):")
        print(f"  Test Acc: {test_results['test_accuracy']:.4f}")
        print(f"  Temporal AUC: {test_results['temporal_auc']:.4f}")
        print(f"  Temporal Acc: {test_results['temporal_accuracy']:.4f}")
        print()
    
    return experiments
