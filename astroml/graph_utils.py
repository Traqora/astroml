"""
Graph utilities for AstroML - PyTorch Geometric conversion functions.
"""

import torch
import numpy as np
from typing import Union, Optional, List
from torch_geometric.data import Data


def graph_to_pyg_data(
    node_features: Union[np.ndarray, List[List[float]]],
    edge_index: Union[np.ndarray, List[List[int]]],
    edge_features: Optional[Union[np.ndarray, List[List[float]]]] = None,
    node_labels: Optional[Union[np.ndarray, List[int]]] = None
) -> Data:
    """
    Convert graph to PyG Data object.
    
    Args:
        node_features: Array of shape [num_nodes, num_node_features]
        edge_index: Array of shape [2, num_edges] or [num_edges, 2]
        edge_features: Optional array of shape [num_edges, num_edge_features]
        node_labels: Optional array of shape [num_nodes]
    
    Returns:
        torch_geometric.data.Data object
    """
    # Convert inputs to numpy arrays if they are lists
    if isinstance(node_features, list):
        node_features = np.array(node_features, dtype=np.float32)
    elif isinstance(node_features, np.ndarray):
        node_features = node_features.astype(np.float32)
    
    if isinstance(edge_index, list):
        edge_index = np.array(edge_index, dtype=np.int64)
    elif isinstance(edge_index, np.ndarray):
        edge_index = edge_index.astype(np.int64)
    
    # Validate node features
    if len(node_features.shape) != 2:
        raise ValueError(f"node_features must be 2D array, got shape {node_features.shape}")
    
    num_nodes = node_features.shape[0]
    
    # Handle edge index format conversion
    if edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
        # Format is [num_edges, 2], need to transpose to [2, num_edges]
        edge_index = edge_index.T
    elif edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, num_edges] or [num_edges, 2], got {edge_index.shape}")
    
    # Validate edge indices are within bounds
    if edge_index.max() >= num_nodes:
        raise ValueError(f"Edge index contains node ID {edge_index.max()} which is >= num_nodes ({num_nodes})")
    if edge_index.min() < 0:
        raise ValueError("Edge index contains negative node IDs")
    
    # Convert to tensors
    x = torch.from_numpy(node_features)  # [num_nodes, num_node_features]
    edge_index_tensor = torch.from_numpy(edge_index)  # [2, num_edges]
    
    # Handle edge features
    edge_attr = None
    if edge_features is not None:
        if isinstance(edge_features, list):
            edge_features = np.array(edge_features, dtype=np.float32)
        elif isinstance(edge_features, np.ndarray):
            edge_features = edge_features.astype(np.float32)
        
        # Validate edge features shape
        if len(edge_features.shape) != 2:
            raise ValueError(f"edge_features must be 2D array, got shape {edge_features.shape}")
        
        if edge_features.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_features shape mismatch: expected {edge_index.shape[1]} edges, "
                f"got {edge_features.shape[0]}"
            )
        
        edge_attr = torch.from_numpy(edge_features)  # [num_edges, num_edge_features]
    
    # Handle node labels
    y = None
    if node_labels is not None:
        if isinstance(node_labels, list):
            node_labels = np.array(node_labels)
        elif isinstance(node_labels, np.ndarray):
            node_labels = node_labels
        
        # Validate node labels shape
        if len(node_labels.shape) > 1:
            raise ValueError(f"node_labels must be 1D array, got shape {node_labels.shape}")
        
        if node_labels.shape[0] != num_nodes:
            raise ValueError(
                f"node_labels shape mismatch: expected {num_nodes} nodes, "
                f"got {node_labels.shape[0]}"
            )
        
        y = torch.from_numpy(node_labels)  # [num_nodes]
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes
    )
    
    return data
