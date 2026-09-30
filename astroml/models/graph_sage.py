"""GraphSAGE node classifier — issue #736.

A drop-in alternative to :class:`astroml.models.gcn.GCN` with the same
constructor shape and the same ``forward(x, edge_index) -> log_softmax``
contract, so anything wired for the GCN can run GraphSAGE by swapping the
class.

Two differences from the existing :class:`astroml.features.gnn.sage.SAGEConv`
matter in practice:

*Aggregation variants* — mean, max, sum and GCN-style aggregation, chosen by
name, rather than mean-only.

*Vectorised aggregation* — messages are scattered into the destination rows
with ``index_add_``/``scatter_reduce_`` in one shot. ``SAGEConv`` loops in
Python over ``dst.unique()``, which is O(V) interpreter iterations per layer
and becomes the bottleneck long before the matmuls do on a real transaction
graph.

This module depends only on ``torch``. Unlike the GCN it does not import
``torch_geometric``, so it stays importable in a deployment that does not
carry PyG.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

#: Aggregation strategies accepted by :class:`SAGEAggregator` and
#: :class:`GraphSAGE`.
AGGREGATIONS = ("mean", "max", "sum", "gcn")


def validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Check that ``edge_index`` really describes edges over ``num_nodes``.

    A silently malformed adjacency is the classic way to train a GNN on
    nonsense: an out-of-range index either crashes deep inside a scatter or,
    worse, wraps to a valid row. Validate once at the boundary instead.

    Returns the tensor as ``long``, ready to index with.
    """
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"edge_index must be a Tensor, got {type(edge_index).__name__}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    edge_index = edge_index.long()
    if edge_index.numel() and num_nodes > 0:
        lowest = int(edge_index.min())
        highest = int(edge_index.max())
        if lowest < 0 or highest >= num_nodes:
            raise ValueError(
                f"edge_index references node {highest if highest >= num_nodes else lowest} "
                f"but the graph has {num_nodes} nodes"
            )
    elif edge_index.numel() and num_nodes == 0:
        raise ValueError("edge_index is non-empty but the graph has no nodes")
    return edge_index


def aggregate_neighbours(
    messages: torch.Tensor,
    destinations: torch.Tensor,
    num_nodes: int,
    aggregation: str,
) -> torch.Tensor:
    """Scatter ``messages`` into per-destination aggregates.

    Args:
        messages: ``[E, F]`` — one message per edge, already gathered from
            the source rows.
        destinations: ``[E]`` — destination node index for each message.
        num_nodes: Number of destination rows to produce.
        aggregation: One of :data:`AGGREGATIONS`.

    Returns:
        ``[num_nodes, F]``. Nodes with no incoming edges are zero for every
        strategy, including ``max`` — an isolated account contributes no
        neighbourhood signal rather than a sentinel.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")

    out = messages.new_zeros((num_nodes, messages.size(-1)))
    if messages.numel() == 0 or num_nodes == 0:
        return out

    if aggregation == "max":
        out = out.scatter_reduce(
            0,
            destinations.unsqueeze(-1).expand_as(messages),
            messages,
            reduce="amax",
            include_self=False,
        )
        # scatter_reduce leaves untouched rows at the init value (zero here),
        # which is what we want, but rows whose only messages are negative
        # would otherwise be clamped to 0. Restore them explicitly.
        touched = torch.zeros(num_nodes, dtype=torch.bool, device=messages.device)
        touched[destinations] = True
        return torch.where(touched.unsqueeze(-1), out, torch.zeros_like(out))

    out = out.index_add(0, destinations, messages)
    if aggregation == "sum":
        return out

    counts = torch.zeros(num_nodes, device=messages.device, dtype=messages.dtype)
    counts = counts.index_add(0, destinations, torch.ones_like(destinations, dtype=messages.dtype))
    if aggregation == "mean":
        # clamp keeps isolated nodes at 0/1 instead of 0/0.
        return out / counts.clamp(min=1.0).unsqueeze(-1)
    # "gcn": the node counts as one of its own neighbours, so the divisor is
    # the neighbourhood size plus one.
    return out / (counts + 1.0).unsqueeze(-1)


class SAGEAggregator(nn.Module):
    """One GraphSAGE layer: aggregate neighbours, then combine with self.

    ``out = W_neigh @ aggregate(neighbours) + W_self @ x``  — the ``gcn``
    variant folds the node into the mean rather than giving it its own
    weight matrix, matching the original paper's GCN aggregator.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregation: str = "mean",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError(f"layer dimensions must be >= 1, got ({in_dim}, {out_dim})")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregation = aggregation

        # The gcn variant folds the node into its own neighbourhood mean, so
        # there is no separate self transform to learn — giving it one would
        # leave a parameter with no gradient, which trips DDP and wastes
        # optimiser state.
        if aggregation == "gcn":
            self.lin_neigh = nn.Linear(in_dim, out_dim, bias=bias)
            self.lin_self: nn.Linear | None = None
        else:
            self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
            self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        if self.lin_neigh.bias is not None:
            nn.init.zeros_(self.lin_neigh.bias)
        if self.lin_self is not None:
            nn.init.xavier_uniform_(self.lin_self.weight)
            if self.lin_self.bias is not None:
                nn.init.zeros_(self.lin_self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        edge_index = validate_edge_index(edge_index, num_nodes)
        src, dst = edge_index[0], edge_index[1]

        aggregated = aggregate_neighbours(x[src], dst, num_nodes, self.aggregation)

        if self.aggregation == "gcn":
            # The self term is already inside the mean; a second weight
            # matrix would double-count it.
            return self.lin_neigh(aggregated + x / (self._neighbour_counts(dst, num_nodes) + 1.0))
        return self.lin_neigh(aggregated) + self.lin_self(x)

    @staticmethod
    def _neighbour_counts(dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        counts = torch.zeros(num_nodes, 1, dtype=torch.float32, device=dst.device)
        if dst.numel():
            counts = counts.index_add(
                0, dst, torch.ones(dst.size(0), 1, dtype=torch.float32, device=dst.device)
            )
        return counts

    def extra_repr(self) -> str:
        return f"{self.in_dim} -> {self.out_dim}, aggregation={self.aggregation}"


class GraphSAGE(nn.Module):
    """GraphSAGE node classifier.

    Mirrors :class:`astroml.models.gcn.GCN`: same positional constructor
    arguments, same ``forward(x, edge_index)`` returning log-probabilities,
    so the two are interchangeable in training and benchmarking code.

    Args:
        input_dim: Node feature width.
        hidden_dim: Width of each hidden layer. Ignored when
            ``hidden_dims`` is given.
        output_dim: Number of classes.
        dropout: Dropout probability applied between layers.
        num_layers: Total number of SAGE layers (>= 1). Two is the usual
            choice; a third rarely helps and triples the neighbourhood a
            single node pulls in.
        aggregator: One of :data:`AGGREGATIONS`.
        hidden_dims: Explicit per-layer hidden widths, e.g. ``[64, 32]``.
            Overrides ``hidden_dim``/``num_layers`` when supplied.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.5,
        num_layers: int = 2,
        aggregator: str = "mean",
        hidden_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        if input_dim < 1 or output_dim < 1:
            raise ValueError(
                f"input_dim and output_dim must be >= 1, got ({input_dim}, {output_dim})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if aggregator not in AGGREGATIONS:
            raise ValueError(f"unknown aggregator {aggregator!r}; expected one of {AGGREGATIONS}")

        if hidden_dims is None:
            if num_layers < 1:
                raise ValueError(f"num_layers must be >= 1, got {num_layers}")
            if hidden_dim < 1:
                raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
            widths = [hidden_dim] * (num_layers - 1)
        else:
            widths = [int(w) for w in hidden_dims]
            if any(w < 1 for w in widths):
                raise ValueError(f"hidden_dims must all be >= 1, got {list(hidden_dims)}")

        self.dropout = dropout
        self.aggregator = aggregator
        self.hidden_dims = tuple(widths)

        dims = [input_dim, *widths, output_dim]
        self.convs = nn.ModuleList(
            SAGEAggregator(dims[i], dims[i + 1], aggregation=aggregator)
            for i in range(len(dims) - 1)
        )

    @property
    def num_layers(self) -> int:
        return len(self.convs)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run every layer and return the raw logits (no softmax).

        Useful when the embeddings themselves are the product — link
        prediction, clustering — rather than the class probabilities.
        """
        if x.dim() != 2:
            raise ValueError(f"x must have shape [N, F], got {tuple(x.shape)}")

        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.embed(x, edge_index), dim=1)


__all__ = [
    "AGGREGATIONS",
    "GraphSAGE",
    "SAGEAggregator",
    "aggregate_neighbours",
    "validate_edge_index",
]
