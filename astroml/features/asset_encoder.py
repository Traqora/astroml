"""Encode asset types as edge features for graph ML models.

This module provides configurable encoding strategies for asset types in
transaction graphs. Asset types can be encoded as one-hot vectors or learned
embeddings, suitable for use as edge features in Graph Neural Networks.
"""
from typing import Dict, List, Literal, Optional

import numpy as np
import torch


class AssetEncoder:
    """Encode asset types as one-hot or embedding vectors.
    
    Args:
        strategy: Encoding strategy - 'onehot' or 'embedding'.
        embedding_dim: Dimension for embedding strategy (ignored for onehot).
    """
    
    def __init__(
        self,
        strategy: Literal['onehot', 'embedding'] = 'onehot',
        embedding_dim: int = 16
    ):
        self.strategy = strategy
        self.embedding_dim = embedding_dim
        self.asset_to_idx: Dict[str, int] = {}
        self.embedding: Optional[torch.nn.Embedding] = None
        
    def fit(self, asset_types: List[str]) -> 'AssetEncoder':
        """Build vocabulary from asset types.
        
        Args:
            asset_types: List of asset type strings.
            
        Returns:
            Self for chaining.
        """
        unique_assets = sorted(set(asset_types))
        self.asset_to_idx = {asset: idx for idx, asset in enumerate(unique_assets)}
        
        if self.strategy == 'embedding':
            self.embedding = torch.nn.Embedding(len(self.asset_to_idx), self.embedding_dim)
            
        return self
    
    def encode(self, asset_types: List[str]) -> np.ndarray:
        """Encode asset types to feature vectors.
        
        Args:
            asset_types: List of asset type strings.
            
        Returns:
            Encoded features as numpy array. Shape is (n_assets, n_classes)
            for one-hot encoding or (n_assets, embedding_dim) for embeddings.
        """
        indices = [self.asset_to_idx.get(a, 0) for a in asset_types]
        
        if self.strategy == 'onehot':
            n_assets = len(self.asset_to_idx)
            encoded = np.zeros((len(indices), n_assets))
            encoded[np.arange(len(indices)), indices] = 1
            return encoded
        else:
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            return self.embedding(indices_tensor).detach().numpy()
    
    @property
    def output_dim(self) -> int:
        """Dimension of encoded features."""
        if self.strategy == 'onehot':
            return len(self.asset_to_idx)
        return self.embedding_dim
