"""Training utilities and advanced Deep SVDD implementations.

This module provides enhanced Deep SVDD training with various loss functions,
optimization strategies, and evaluation metrics for fraud detection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from .deep_svdd import DeepSVDD, DeepSVDDNetwork
from astroml.tracking import MLflowTracker


class DeepSVDDTrainer:
    """Advanced trainer for Deep SVDD with multiple loss functions and strategies."""
    
    def __init__(
        self,
        model: DeepSVDD,
        device: str = 'cpu',
        patience: int = 10,
        min_delta: float = 1e-4,
        tracker: Optional[MLflowTracker] = None,
    ):
        self.model = model
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.tracker = tracker  # None → no MLflow logging

        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'radius': []
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        loss_type: str = 'svdd',
        scheduler_type: str = 'cosine'
    ) -> Dict[str, np.ndarray]:
        """Train Deep SVDD with advanced strategies."""
        
        # Initialize center
        self.model.init_center(train_loader)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        scheduler = self._get_scheduler(optimizer, scheduler_type, epochs)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, loss_type)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, loss_type)
                
                # Early stopping
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Update scheduler
            if scheduler_type != 'none':
                scheduler.step()
            
            # Update radius
            radius = self._compute_radius(train_loader)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            self.training_history['radius'].append(radius)

            # MLflow per-epoch metrics
            if self.tracker is not None:
                step_metrics: Dict[str, float] = {
                    "train_loss": train_loss,
                    "svdd_radius": radius,
                }
                if val_loss is not None:
                    step_metrics["val_loss"] = val_loss
                self.tracker.log_metrics(step_metrics, step=epoch)

            # Console logging
            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss = {val_loss:.4f}"
                log_msg += f", Radius = {radius:.4f}"
                print(log_msg)

        return self.training_history
    
    def _train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_type: str
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            
            optimizer.zero_grad()
            
            if loss_type == 'svdd':
                loss = self.model.compute_loss(x)
            elif loss_type == 'soft_boundary':
                loss = self._soft_boundary_loss(x)
            elif loss_type == 'robust':
                loss = self._robust_loss(x)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_type: str
    ) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                if loss_type == 'svdd':
                    loss = self.model.compute_loss(x)
                elif loss_type == 'soft_boundary':
                    loss = self._soft_boundary_loss(x)
                elif loss_type == 'robust':
                    loss = self._robust_loss(x)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _soft_boundary_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Soft boundary loss for more flexible anomaly detection."""
        embeddings = self.model(x)
        distances = torch.sum((embeddings - self.model.center) ** 2, dim=1)
        
        # Soft boundary with radius R
        radius = self._compute_radius_single_batch(x)
        loss = torch.mean(torch.relu(distances - radius))
        
        return loss
    
    def _robust_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Robust loss function less sensitive to outliers."""
        embeddings = self.model(x)
        distances = torch.sum((embeddings - self.model.center) ** 2, dim=1)
        
        # Huber-like loss
        delta = torch.median(distances)
        loss = torch.where(
            distances <= delta,
            0.5 * distances,
            delta * (torch.sqrt(distances) - 0.5 * torch.sqrt(delta))
        )
        
        return torch.mean(loss)
    
    def _compute_radius(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Compute hypersphere radius."""
        self.model.eval()
        all_distances = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                embeddings = self.model(x)
                distances = torch.sum((embeddings - self.model.center) ** 2, dim=1)
                all_distances.append(distances)
        
        all_distances = torch.cat(all_distances, dim=0)
        
        # Set radius to capture (1-nu) quantile of normal data
        radius = torch.quantile(all_distances, 1 - self.model.nu)
        return radius.item()
    
    def _compute_radius_single_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Compute radius for a single batch."""
        embeddings = self.model(x)
        distances = torch.sum((embeddings - self.model.center) ** 2, dim=1)
        return torch.quantile(distances, 1 - self.model.nu)
    
    def _get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        epochs: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Get learning rate scheduler."""
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=epochs // 3, gamma=0.1
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
        elif scheduler_type == 'none':
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: 1.0
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _save_checkpoint(self):
        """Save best model checkpoint and log it to MLflow."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'center': self.model.center,
            'scaler': self.model.scaler if hasattr(self.model, 'scaler') else None,
            'training_history': self.training_history
        }
        torch.save(checkpoint, 'best_deep_svdd.pth')
        if self.tracker is not None:
            self.tracker.log_model_artifact(
                self.model,
                artifact_path="model",
                checkpoint_path="best_deep_svdd.pth",
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.center = checkpoint['center']
        
        if checkpoint.get('scaler') is not None:
            self.model.scaler = checkpoint['scaler']
        
        if checkpoint.get('training_history') is not None:
            self.training_history = checkpoint['training_history']
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold_percentile: float = 95.0,
    ) -> Dict[str, float]:
        """Evaluate model performance and log results to MLflow."""

        # Get anomaly scores
        scores = self.model.predict(X)

        # Determine threshold
        threshold = np.percentile(scores, threshold_percentile)
        predictions = (scores > threshold).astype(int)

        # Calculate metrics
        metrics: Dict[str, float] = {}

        # AUC-ROC
        try:
            metrics['auc_roc'] = roc_auc_score(y, scores)
        except ValueError:
            metrics['auc_roc'] = 0.0

        # AUC-PR
        try:
            precision, recall, _ = precision_recall_curve(y, scores)
            metrics['auc_pr'] = auc(recall, precision)
        except ValueError:
            metrics['auc_pr'] = 0.0

        # Basic classification metrics
        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        tn = np.sum((predictions == 0) & (y == 0))

        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall']
        ) if (metrics['precision'] + metrics['recall']) > 0 else 0
        metrics['accuracy'] = (tp + tn) / (tp + fp + fn + tn)

        # Log evaluation metrics (ROC-AUC, precision, recall, f1, accuracy)
        if self.tracker is not None:
            self.tracker.log_metrics({
                "eval_roc_auc": metrics['auc_roc'],
                "eval_auc_pr": metrics['auc_pr'],
                "eval_precision": metrics['precision'],
                "eval_recall": metrics['recall'],
                "eval_f1": metrics['f1'],
                "eval_accuracy": metrics['accuracy'],
            })

        return metrics


class FraudDetectionDeepSVDD:
    """Specialized Deep SVDD for fraud detection with domain-specific features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64, 32],
        dropout: float = 0.2,
        nu: float = 0.05,  # Lower nu for fraud detection (few anomalies)
        device: str = 'cpu'
    ):
        self.model = DeepSVDD(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            nu=nu,
            device=device
        )
        self.trainer = DeepSVDDTrainer(self.model, device=device)
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        validation_split: float = 0.2,
        **training_kwargs
    ) -> 'FraudDetectionDeepSVDD':
        """Fit model for fraud detection."""
        
        # Split data for validation
        if validation_split > 0:
            n_samples = len(X)
            val_size = int(n_samples * validation_split)
            indices = np.random.permutation(n_samples)
            
            train_idx, val_idx = indices[val_size:], indices[:val_size]
            X_train, X_val = X[train_idx], X[val_idx]
            
            if y is not None:
                y_train, y_val = y[train_idx], y[val_idx]
            else:
                y_train, y_val = None, None
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True
        )
        
        val_loader = None
        if X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val)
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=128, shuffle=False
            )
        
        # Train model
        self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            **training_kwargs
        )
        
        return self
    
    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for transactions."""
        return self.model.predict(X)
    
    def predict_fraud_probability(self, X: np.ndarray) -> np.ndarray:
        """Convert anomaly scores to fraud probabilities."""
        scores = self.predict_anomaly_scores(X)
        
        # Normalize scores to [0, 1] using min-max scaling
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            probabilities = (scores - min_score) / (max_score - min_score)
        else:
            probabilities = np.zeros_like(scores)
        
        return probabilities
    
    def evaluate_fraud_detection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold_percentile: float = 95.0
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Evaluate fraud detection performance."""
        
        metrics = self.trainer.evaluate(X, y, threshold_percentile)
        scores = self.predict_anomaly_scores(X)
        
        return {
            **metrics,
            'anomaly_scores': scores,
            'fraud_probabilities': self.predict_fraud_probability(X)
        }
