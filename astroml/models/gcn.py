from typing import Any, Dict, List, Optional, Union, Callable
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Standard 2-layer Graph Convolutional Network for node classification.

    Architecture: GCNConv -> ReLU -> Dropout -> GCNConv -> log_softmax
    """

    def __init__(self, input_dim -> Any: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index) -> Any:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
