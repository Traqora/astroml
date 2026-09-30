import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, MessagePassing, SAGEConv
from torch_geometric.utils import degree


class TemporalEncoding(nn.Module):
    """Temporal encoding using sinusoidal functions."""

    def __init__(self, temporal_dim -> Any: int, max_time: float = 1000.0):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.max_time = max_time

        # Create sinusoidal encoding
        position = torch.arange(0, temporal_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, temporal_dim, 2).float() * (-math.log(10000.0) / temporal_dim)
        )

        self.register_buffer("encoding", torch.zeros(temporal_dim))
        self.encoding[0::2] = torch.sin(position / div_term)
        self.encoding[1::2] = torch.cos(position / div_term)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps to temporal features.

        Args:
            timestamps: Tensor of shape [num_nodes] with timestamp values

        Returns:
            Temporal encoding tensor of shape [num_nodes, temporal_dim]
        """
        # Normalize timestamps to [0, 1]
        normalized_time = timestamps / self.max_time

        # Scale to encoding dimension
        scaled_time = normalized_time * self.temporal_dim

        # Use sinusoidal encoding
        indices = scaled_time.long() % self.temporal_dim
        return self.encoding[indices]


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time-aware node representations."""

    def __init__(self, input_dim -> Any: int, temporal_dim: int, heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.heads = heads
        self.head_dim = input_dim // heads

        assert input_dim % heads == 0, "input_dim must be divisible by heads"

        # Query, Key, Value projections
        self.query = nn.Linear(input_dim + temporal_dim, input_dim)
        self.key = nn.Linear(input_dim + temporal_dim, input_dim)
        self.value = nn.Linear(input_dim + temporal_dim, input_dim)

        # Output projection
        self.out_proj = nn.Linear(input_dim, input_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, temporal_encoding: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention.

        Args:
            x: Node features [num_nodes, input_dim]
            temporal_encoding: Temporal features [num_nodes, temporal_dim]

        Returns:
            Attention-enhanced features [num_nodes, input_dim]
        """
        batch_size = x.size(0)

        # Combine features with temporal encoding
        combined = torch.cat([x, temporal_encoding], dim=-1)

        # Compute Q, K, V
        Q = self.query(combined).view(batch_size, self.heads, self.head_dim)
        K = self.key(combined).view(batch_size, self.heads, self.head_dim)
        V = self.value(combined).view(batch_size, self.heads, self.head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attended = torch.matmul(attn_weights, V)
        attended = attended.view(batch_size, self.input_dim)

        # Output projection
        output = self.out_proj(attended)

        return output + x  # Residual connection


class TemporalGCN(nn.Module):
    """Temporal Graph Convolutional Network with time encoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temporal_dim: int = 32,
        dropout: float = 0.5,
        time_encoding: str = "sinusoidal",
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.temporal_dim = temporal_dim
        self.dropout = dropout
        self.time_encoding = time_encoding
        self.use_attention = use_attention

        # Temporal encoding
        if time_encoding == "sinusoidal":
            self.temporal_encoder = TemporalEncoding(temporal_dim)
        elif time_encoding == "learnable":
            self.temporal_encoder = nn.Linear(1, temporal_dim)
        else:
            raise ValueError(f"Unknown time encoding: {time_encoding}")

        # Adjust input dimension for temporal features
        effective_input_dim = input_dim + temporal_dim

        # Graph convolution layers
        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(effective_input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))

        # Output layer
        self.convs.append(GCNConv(hidden_dims[-1], output_dim))

        # Temporal attention
        if use_attention:
            self.temporal_attention = TemporalAttention(hidden_dims[-1], temporal_dim, heads=8)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: Optional[torch.Tensor] = None,
        node_time: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with temporal information.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_time: Edge timestamps [num_edges] (optional)
            node_time: Node timestamps [num_nodes] (optional)
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional)

        Returns:
            Node predictions [num_nodes, output_dim]
        """
        # Default timestamps if not provided
        if node_time is None:
            node_time = torch.zeros(x.size(0), device=x.device)

        # Encode temporal information
        if self.time_encoding == "learnable":
            temporal_encoding = self.temporal_encoder(node_time.unsqueeze(-1))
        else:
            temporal_encoding = self.temporal_encoder(node_time)

        # Combine features with temporal encoding
        x_combined = torch.cat([x, temporal_encoding], dim=-1)

        # Apply graph convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x_combined = conv(x_combined, edge_index, edge_attr)
            x_combined = F.relu(x_combined)
            x_combined = self.dropout_layer(x_combined)

        # Output layer
        x_out = self.convs[-1](x_combined, edge_index, edge_attr)

        # Apply temporal attention if enabled
        if self.use_attention and hasattr(self, "temporal_attention"):
            x_out = self.temporal_attention(x_out, temporal_encoding)

        return F.log_softmax(x_out, dim=1)


class TemporalGraphSAGE(nn.Module):
    """Temporal GraphSAGE with temporal encoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temporal_dim: int = 32,
        dropout: float = 0.5,
        num_layers: int = 2,
        aggregator: str = "mean",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.temporal_dim = temporal_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.aggregator = aggregator

        # Temporal encoding
        self.temporal_encoder = TemporalEncoding(temporal_dim)

        # Adjust input dimension for temporal features
        effective_input_dim = input_dim + temporal_dim

        # GraphSAGE layers
        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(effective_input_dim, hidden_dims[0], aggr=aggregator))

        # Hidden layers
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dims[i], hidden_dims[i + 1], aggr=aggregator))

        # Output layer
        self.convs.append(SAGEConv(hidden_dims[-1], output_dim, aggr=aggregator))

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with temporal information.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            node_time: Node timestamps [num_nodes] (optional)

        Returns:
            Node predictions [num_nodes, output_dim]
        """
        # Default timestamps if not provided
        if node_time is None:
            node_time = torch.zeros(x.size(0), device=x.device)

        # Encode temporal information
        temporal_encoding = self.temporal_encoder(node_time)

        # Combine features with temporal encoding
        x_combined = torch.cat([x, temporal_encoding], dim=-1)

        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs[:-1]):
            x_combined = conv(x_combined, edge_index)
            x_combined = F.relu(x_combined)
            x_combined = self.dropout_layer(x_combined)

        # Output layer
        x_out = self.convs[-1](x_combined, edge_index)

        return F.log_softmax(x_out, dim=1)


class TemporalGAT(nn.Module):
    """Temporal Graph Attention Network with temporal encoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temporal_dim: int = 32,
        dropout: float = 0.5,
        heads: int = 8,
        concat: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.temporal_dim = temporal_dim
        self.dropout = dropout
        self.heads = heads
        self.concat = concat

        # Temporal encoding
        self.temporal_encoder = TemporalEncoding(temporal_dim)

        # Adjust input dimension for temporal features
        effective_input_dim = input_dim + temporal_dim

        # GAT layers
        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(
            GATConv(
                effective_input_dim,
                hidden_dims[0] // heads,
                heads=heads,
                dropout=dropout,
                concat=concat,
            )
        )

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(
                GATConv(
                    hidden_dims[i],
                    hidden_dims[i + 1] // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=concat,
                )
            )

        # Output layer (no multi-head for final layer)
        self.convs.append(
            GATConv(hidden_dims[-1], output_dim, heads=1, dropout=dropout, concat=False)
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_time: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with temporal information.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            node_time: Node timestamps [num_nodes] (optional)
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional)

        Returns:
            Node predictions [num_nodes, output_dim]
        """
        # Default timestamps if not provided
        if node_time is None:
            node_time = torch.zeros(x.size(0), device=x.device)

        # Encode temporal information
        temporal_encoding = self.temporal_encoder(node_time)

        # Combine features with temporal encoding
        x_combined = torch.cat([x, temporal_encoding], dim=-1)

        # Apply GAT layers
        for i, conv in enumerate(self.convs[:-1]):
            x_combined = conv(x_combined, edge_index, edge_attr)
            x_combined = F.elu(x_combined)  # GAT typically uses ELU
            x_combined = self.dropout_layer(x_combined)

        # Output layer
        x_out = self.convs[-1](x_combined, edge_index, edge_attr)

        return F.log_softmax(x_out, dim=1)


class TemporalEdgeConv(MessagePassing):
    """Temporal edge convolution for dynamic edge features."""

    def __init__(
        self, in_channels: int, out_channels: int, temporal_dim: int = 32, aggr: str = "max"
    ):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_dim = temporal_dim

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + temporal_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_time: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for temporal edge convolution.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_time: Edge timestamps [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Temporal encoding for edges
        temporal_encoder = TemporalEncoding(self.temporal_dim)
        edge_temporal_encoding = temporal_encoder(edge_time)

        # Propagate messages
        return self.propagate(edge_index, x=x, edge_temporal=edge_temporal_encoding)

    def message(
        self, x_j: torch.Tensor, x_i: torch.Tensor, edge_temporal: torch.Tensor
    ) -> torch.Tensor:
        """Message function for edge convolution."""
        # Combine source, target, and temporal features
        combined = torch.cat([x_i, x_j, edge_temporal], dim=-1)
        return self.message_mlp(combined)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update function for edge convolution."""
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(combined)


class TemporalGraphTransformer(nn.Module):
    """Temporal Graph Transformer with attention mechanisms."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        temporal_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.temporal_dim = temporal_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Temporal encoding
        self.temporal_encoder = TemporalEncoding(temporal_dim)

        # Input projection
        self.input_proj = nn.Linear(input_dim + temporal_dim, hidden_dim)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(1000, hidden_dim)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_time: Optional[torch.Tensor] = None,
        edge_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for temporal graph transformer.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            node_time: Node timestamps [num_nodes] (optional)
            edge_time: Edge timestamps [num_edges] (optional)

        Returns:
            Node predictions [num_nodes, output_dim]
        """
        # Default timestamps if not provided
        if node_time is None:
            node_time = torch.zeros(x.size(0), device=x.device)

        # Encode temporal information
        temporal_encoding = self.temporal_encoder(node_time)

        # Combine features with temporal encoding
        x_combined = torch.cat([x, temporal_encoding], dim=-1)

        # Input projection
        x_proj = self.input_proj(x_combined)

        # Add positional encoding
        seq_len = x_proj.size(0)
        pos_encoding = self.pos_encoding[:, :seq_len, :].to(x_proj.device)
        x_proj = x_proj + pos_encoding.squeeze(0)

        # Apply transformer layers
        x_transformed = x_proj.unsqueeze(0)  # Add batch dimension
        for layer in self.transformer_layers:
            x_transformed = layer(x_transformed)

        x_transformed = x_transformed.squeeze(0)  # Remove batch dimension

        # Output projection
        x_out = self.output_proj(x_transformed)

        return F.log_softmax(x_out, dim=1)


class TemporalModelFactory:
    """Factory for creating temporal GNN models."""

    @staticmethod
    def create_temporal_gcn(config) -> TemporalGCN:
        """Create TemporalGCN model."""
        return TemporalGCN(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            temporal_dim=config.get("temporal_dim", 32),
            dropout=config.get("dropout", 0.5),
            time_encoding=config.get("time_encoding", "sinusoidal"),
            use_attention=config.get("use_attention", True),
        )

    @staticmethod
    def create_temporal_sage(config) -> TemporalGraphSAGE:
        """Create TemporalGraphSAGE model."""
        return TemporalGraphSAGE(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            temporal_dim=config.get("temporal_dim", 32),
            dropout=config.get("dropout", 0.5),
            num_layers=config.get("num_layers", 2),
            aggregator=config.get("aggregator", "mean"),
        )

    @staticmethod
    def create_temporal_gat(config) -> TemporalGAT:
        """Create TemporalGAT model."""
        return TemporalGAT(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            temporal_dim=config.get("temporal_dim", 32),
            dropout=config.get("dropout", 0.5),
            heads=config.get("heads", 8),
            concat=config.get("concat", True),
        )

    @staticmethod
    def create_temporal_transformer(config) -> TemporalGraphTransformer:
        """Create TemporalGraphTransformer model."""
        return TemporalGraphTransformer(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            temporal_dim=config.get("temporal_dim", 32),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.1),
        )

    @staticmethod
    def create_tgn(config) -> Any:
        """Create a TemporalGraphNetwork (issue #737).

        Imported here rather than at module scope: tgn.py depends only on
        torch, and importing it from a module that already requires
        torch_geometric would tie it to a dependency it does not need.
        """
        from .tgn import TemporalGraphNetwork

        return TemporalGraphNetwork(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            memory_dim=config.get("memory_dim", 64),
            time_dim=config.get("temporal_dim", 32),
            edge_dim=config.get("edge_dim", 0),
            message_dim=config.get("message_dim", 64),
            num_nodes=config.get("num_nodes", None),
            dropout=config.get("dropout", 0.1),
            message_aggregation=config.get("message_aggregation", "mean"),
        )
