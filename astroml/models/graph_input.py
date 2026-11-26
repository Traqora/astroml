"""Validated model input for the baseline graph model (issue #1200).

The baseline :class:`~astroml.models.gcn.GCN` takes node features ``x`` and a COO
``edge_index``. Graphs built from real ledger data routinely contain nodes with no
edges (an account that has only received one payment inside the window, or a
snapshot with no payments at all), and the raw tensors fail in unhelpful ways on
those inputs:

* An empty edge list written as ``torch.tensor([[], []])`` is *float*, and the
  scatter kernel rejects it with ``Expected dtype int32/int64 for index``.
* An out-of-range node index surfaces as a bare ``IndexError`` from deep inside
  the model.
* A ``NaN`` feature does not raise at all: it propagates to the output.

:func:`prepare_graph_input` checks the tensors once, up front, and returns them in
a form the model accepts together with a count of isolated nodes.

Isolated nodes are supported, not an error
------------------------------------------
:class:`~astroml.models.gcn.GCN` adds a self-loop to every node, so a node with
no neighbours is represented by its own features alone. Its output is identical
to the output it would have as a one-node graph. A graph with no edges at all is
valid input for the same reason. Self-loops already present in ``edge_index`` do
not connect a node to anything, so a node whose only edge is a self-loop counts
as isolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["GraphInput", "isolated_node_mask", "prepare_graph_input"]


@dataclass(frozen=True)
class GraphInput:
    """Validated tensors ready for the model, plus a summary of the graph.

    Attributes:
        x: Node features, shape ``[N, F]``, floating point, all values finite.
        edge_index: COO edges, shape ``[2, E]``, dtype ``torch.long``. ``E`` may be
            zero.
        num_nodes: ``N``.
        num_edges: ``E``.
        num_isolated: Nodes with no edge to a different node.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int
    num_edges: int
    num_isolated: int

    @property
    def isolated_fraction(self) -> float:
        """Share of nodes that are isolated; ``0.0`` for an empty graph."""
        return self.num_isolated / self.num_nodes if self.num_nodes else 0.0


def isolated_node_mask(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Boolean mask, ``True`` for nodes with no edge to a different node.

    Args:
        edge_index: COO edges, shape ``[2, E]``, integer dtype. May be empty.
        num_nodes: Number of nodes the indices address.

    Returns:
        A ``[num_nodes]`` bool tensor on ``edge_index``'s device.
    """
    connected = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    if edge_index.numel() == 0:
        return ~connected

    source, target = edge_index[0], edge_index[1]
    linked = source != target
    connected[source[linked]] = True
    connected[target[linked]] = True
    return ~connected


def prepare_graph_input(x: torch.Tensor, edge_index: torch.Tensor) -> GraphInput:
    """Validate model input and normalise an empty edge list.

    Args:
        x: Node features, shape ``[N, F]``, floating point.
        edge_index: COO edges, shape ``[2, E]``. An empty edge list of any dtype
            is accepted and returned as ``torch.long``; a non-empty one must
            already be an integer tensor.

    Returns:
        The validated tensors and a summary that includes the isolated-node count.

    Raises:
        ValueError: If ``x`` is not a finite floating-point matrix, if
            ``edge_index`` is not shaped ``[2, E]`` or holds non-integer values, or
            if any index falls outside ``[0, N)``. Each message names the tensor and
            the offending value.
    """
    if x.dim() != 2:
        raise ValueError(f"x must have shape [N, F], got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise ValueError(f"x must be floating point, got {x.dtype}")
    if x.numel() and not bool(torch.isfinite(x).all()):
        bad = int((~torch.isfinite(x).all(dim=1)).nonzero()[0])
        raise ValueError(f"x contains NaN or infinite values (first bad row: node {bad})")

    num_nodes = int(x.size(0))

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    if edge_index.numel() == 0:
        # ``torch.tensor([[], []])`` is float32; an empty edge list has no
        # indices to be wrong, so it is safe to retype.
        edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    elif edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"edge_index must hold integer indices, got {edge_index.dtype}")
    else:
        edge_index = edge_index.to(torch.long)
        lowest, highest = int(edge_index.min()), int(edge_index.max())
        if lowest < 0 or highest >= num_nodes:
            raise ValueError(
                f"edge_index has an index outside [0, {num_nodes}): " f"min={lowest}, max={highest}"
            )

    return GraphInput(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_edges=int(edge_index.size(1)),
        num_isolated=int(isolated_node_mask(edge_index, num_nodes).sum()),
    )
