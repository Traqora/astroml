"""Deep Support Vector Data Description for unsupervised anomaly detection.

Deep SVDD learns a hypersphere that encloses normal data points in feature space,
with anomalies falling outside this boundary. Ideal for fraud detection with
limited labeled examples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class DeepSVDDNetwork(nn.Module):
    """Neural network for Deep SVDD feature extraction."""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DeepSVDD(nn.Module, BaseEstimator):
    """Deep Support Vector Data Description for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.1,
        nu: float = 0.1,
        center: Optional[torch.Tensor] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.nu = nu
        self.device = device
        
        self.network = DeepSVDDNetwork(input_dim, hidden_dims, dropout)
        self.center = center
        
        self.to(device)
    
    def init_center(self, data_loader: torch.utils.data.DataLoader):
        """Initialize hypersphere center from normal data."""
        self.eval()
        with torch.no_grad():
            embeddings = []
            for x, _ in data_loader:
                x = x.to(self.device).float()
                embedding = self.network(x)
                embeddings.append(embedding)
            
            all_embeddings = torch.cat(embeddings, dim=0)
            self.center = torch.mean(all_embeddings, dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Deep SVDD loss."""
        embeddings = self.network(x)
        distances = torch.sum((embeddings - self.center) ** 2, dim=1)
        return torch.mean(distances)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict anomaly scores."""
        from astroml.observability.metrics import MODEL_INFERENCE_LATENCY

        self.eval()
        # Model inference latency (issue #567).
        with MODEL_INFERENCE_LATENCY.labels("deep_svdd").time(), torch.no_grad():
            x = x.to(self.device).float()
            embeddings = self.network(x)
            distances = torch.sum((embeddings - self.center) ** 2, dim=1)
            return distances.cpu().numpy()
    
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 128,
        lr: float = 0.001
    ) -> 'DeepSVDD':
        """Fit Deep SVDD model."""
        
        # Scale data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_scaled)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Initialize center
        self.init_center(dataloader)
        
        # Training
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            for batch, in dataloader:
                batch = batch[0].to(self.device)
                
                optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        return self
