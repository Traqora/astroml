"""Temporal Graph Network (TGN) — issue #737.

The temporal models already in :mod:`astroml.models.temporal` are static GNNs
with a time encoding bolted onto the node features: each snapshot is scored
independently and nothing carries over between them. TGN keeps a **memory**
vector per node that is updated by every interaction the node takes part in,
so a snapshot is read in the context of everything that came before it.

The pieces, following Rossi et al. (2020):

* :class:`TimeEncoder` — the learnable Bochner encoding ``cos(w·Δt + b)``,
  applied to the *elapsed* time since a node was last touched rather than to
  an absolute timestamp, which is what makes the model transferable across
  windows.
* :class:`MessageFunction` — builds a message per interaction from both
  endpoints' memories, the elapsed time and any edge features.
* :class:`MemoryUpdater` — a GRU cell folding aggregated messages into a
  node's memory.
* :class:`TemporalGraphNetwork` — the model, exposing the repository's
  standard ``forward(x, edge_index, edge_time=..., node_time=..., edge_attr=...)``
  returning log-probabilities, so it drops into the same training and
  benchmarking code as :class:`~astroml.models.temporal.TemporalGCN`, plus
  :meth:`~TemporalGraphNetwork.forward_stream` for consuming a snapshot
  sequence.

**Determinism.** Interactions inside a snapshot are sorted by
``(timestamp, source, destination)`` before the memory is touched, so a batch
of events that arrived from the database in an arbitrary order updates memory
in one fixed sequence. Message aggregation is a scatter-add rather than a
Python loop over unique destinations, which is both faster and reproducible.
Memory is explicit state: :meth:`~TemporalGraphNetwork.reset_memory` puts the
model back to a known start, and training must call it between epochs or the
second epoch starts from the first one's leftovers.

**Cost.** Memory is O(N × memory_dim) and updates are batched — one GRU call
per snapshot, not one per edge — so a snapshot costs O(E) message
construction plus O(N) update, with no dependence on how much history the
memory summarises.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

#: How the messages arriving at one node in a batch are combined.
MESSAGE_AGGREGATIONS = ("mean", "sum", "last")


class TimeEncoder(nn.Module):
    """Learnable time encoding, ``cos(w·Δt + b)``.

    Args:
        dimension: Width of the encoding.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        self.dimension = dimension
        self.basis = nn.Linear(1, dimension)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Geometrically spaced frequencies span many timescales at once, so
        # the encoding separates "seconds ago" from "days ago" without having
        # to learn the scale from scratch.
        with torch.no_grad():
            frequencies = 1.0 / (10.0 ** torch.linspace(0, 9, self.dimension))
            self.basis.weight.copy_(frequencies.reshape(self.dimension, 1))
            self.basis.bias.zero_()

    def forward(self, elapsed: torch.Tensor) -> torch.Tensor:
        """Encode elapsed times.

        Args:
            elapsed: ``[E]`` or ``[E, 1]`` non-negative elapsed times.

        Returns:
            ``[E, dimension]``.
        """
        if elapsed.dim() == 1:
            elapsed = elapsed.unsqueeze(-1)
        return torch.cos(self.basis(elapsed.float()))


class MessageFunction(nn.Module):
    """Builds one message per interaction.

    The raw message concatenates the source memory, the destination memory,
    the encoded elapsed time and the edge features; a linear layer projects
    it to ``message_dim``.
    """

    def __init__(
        self,
        memory_dim: int,
        time_dim: int,
        edge_dim: int,
        message_dim: int,
    ) -> None:
        super().__init__()
        self.raw_dim = 2 * memory_dim + time_dim + edge_dim
        self.message_dim = message_dim
        self.projection = nn.Linear(self.raw_dim, message_dim)

    def forward(
        self,
        source_memory: torch.Tensor,
        destination_memory: torch.Tensor,
        time_encoding: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        raw = torch.cat([source_memory, destination_memory, time_encoding, edge_features], dim=-1)
        return F.relu(self.projection(raw))


class MemoryUpdater(nn.Module):
    """Folds aggregated messages into node memory with a GRU cell."""

    def __init__(self, message_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(message_dim, memory_dim)

    def forward(self, messages: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.cell(messages, memory)


@dataclass
class MemoryState:
    """The model's per-node state between snapshots."""

    memory: torch.Tensor
    last_update: torch.Tensor

    def detach(self) -> MemoryState:
        """Cut the autograd graph while keeping the values.

        Called between snapshots so backpropagation does not reach back
        through every snapshot ever processed — without it the graph grows
        without bound and the first epoch eventually runs out of memory.
        """
        return MemoryState(self.memory.detach(), self.last_update.detach())

    def clone(self) -> MemoryState:
        return MemoryState(self.memory.clone(), self.last_update.clone())


def aggregate_messages(
    messages: torch.Tensor,
    destinations: torch.Tensor,
    num_nodes: int,
    aggregation: str = "mean",
) -> torch.Tensor:
    """Scatter per-interaction messages into per-node aggregates.

    Args:
        messages: ``[E, message_dim]``.
        destinations: ``[E]`` node index each message is addressed to.
        num_nodes: Rows to produce.
        aggregation: One of :data:`MESSAGE_AGGREGATIONS`. ``"last"`` keeps
            only the final message per node in the order given, which is why
            callers sort their interactions first.

    Returns:
        ``[num_nodes, message_dim]``; nodes with no messages are zero.
    """
    if aggregation not in MESSAGE_AGGREGATIONS:
        raise ValueError(
            f"unknown aggregation {aggregation!r}; expected one of {MESSAGE_AGGREGATIONS}"
        )

    out = messages.new_zeros((num_nodes, messages.size(-1)))
    if messages.numel() == 0 or num_nodes == 0:
        return out

    if aggregation == "last":
        # index_copy honours the order of the index tensor, so the final
        # message for a node wins — deterministic because the caller sorted.
        return out.index_copy(0, destinations, messages)

    out = out.index_add(0, destinations, messages)
    if aggregation == "sum":
        return out

    counts = torch.zeros(num_nodes, device=messages.device, dtype=messages.dtype)
    counts = counts.index_add(0, destinations, torch.ones_like(destinations, dtype=messages.dtype))
    return out / counts.clamp(min=1.0).unsqueeze(-1)


class TemporalGraphNetwork(nn.Module):
    """TGN-style temporal model over a dynamic snapshot stream.

    Args:
        input_dim: Node feature width.
        hidden_dim: Width of the embedding head.
        output_dim: Number of classes.
        memory_dim: Width of the per-node memory.
        time_dim: Width of the time encoding.
        edge_dim: Width of ``edge_attr``. Zero means edges carry no features.
        message_dim: Width of an interaction message.
        num_nodes: Size of the memory table. May be omitted and inferred from
            the first batch, but pin it for training so memory survives
            batches that do not mention every node.
        dropout: Dropout in the embedding head.
        message_aggregation: One of :data:`MESSAGE_AGGREGATIONS`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        memory_dim: int = 64,
        time_dim: int = 32,
        edge_dim: int = 0,
        message_dim: int = 64,
        num_nodes: int | None = None,
        dropout: float = 0.1,
        message_aggregation: str = "mean",
    ) -> None:
        super().__init__()

        if input_dim < 1 or output_dim < 1:
            raise ValueError(
                f"input_dim and output_dim must be >= 1, got ({input_dim}, {output_dim})"
            )
        if hidden_dim < 1 or memory_dim < 1 or message_dim < 1:
            raise ValueError("hidden_dim, memory_dim and message_dim must all be >= 1")
        if edge_dim < 0:
            raise ValueError(f"edge_dim must be >= 0, got {edge_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if message_aggregation not in MESSAGE_AGGREGATIONS:
            raise ValueError(
                f"unknown message_aggregation {message_aggregation!r}; "
                f"expected one of {MESSAGE_AGGREGATIONS}"
            )
        if num_nodes is not None and num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1 or None, got {num_nodes}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        self.message_dim = message_dim
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.message_aggregation = message_aggregation

        self.time_encoder = TimeEncoder(time_dim)
        self.message_function = MessageFunction(memory_dim, time_dim, edge_dim, message_dim)
        self.memory_updater = MemoryUpdater(message_dim, memory_dim)

        # Embedding head: the node's own features alongside its memory.
        self.embedding = nn.Sequential(
            nn.Linear(input_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._state: MemoryState | None = None
        if num_nodes is not None:
            self.reset_memory(num_nodes)

    # -- memory ----------------------------------------------------------

    @property
    def memory_initialised(self) -> bool:
        return self._state is not None

    def reset_memory(self, num_nodes: int | None = None, device: Any = None) -> MemoryState:
        """Clear memory back to zeros.

        Call between epochs: memory is state, and an epoch that starts from
        the previous one's leftovers is not the run the config describes.
        """
        size = num_nodes if num_nodes is not None else self.num_nodes
        if size is None:
            raise ValueError("num_nodes is unknown; pass it here or to the constructor")
        if device is None:
            device = next(self.parameters()).device

        self.num_nodes = size
        self._state = MemoryState(
            memory=torch.zeros(size, self.memory_dim, device=device),
            last_update=torch.zeros(size, device=device),
        )
        return self._state

    def get_memory(self) -> MemoryState:
        if self._state is None:
            raise RuntimeError("memory has not been initialised; call reset_memory() first")
        return self._state

    def detach_memory(self) -> None:
        """Detach memory from the autograd graph between snapshots."""
        if self._state is not None:
            self._state = self._state.detach()

    def _ensure_memory(self, num_nodes: int, device: Any) -> MemoryState:
        if self._state is None:
            return self.reset_memory(num_nodes, device=device)
        if self._state.memory.size(0) < num_nodes:
            raise ValueError(
                f"memory holds {self._state.memory.size(0)} nodes but the batch "
                f"references {num_nodes}; construct with a larger num_nodes"
            )
        return self._state

    # -- the temporal update --------------------------------------------

    @staticmethod
    def _canonical_order(
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
    ) -> torch.Tensor:
        """Indices sorting interactions by (time, src, dst).

        Memory updates are order-dependent, so a snapshot whose rows came back
        from the database in an arbitrary order must be put into one fixed
        sequence before anything is written.
        """
        source, destination = edge_index[0], edge_index[1]
        # Stable sorts applied least-significant key first compose into a
        # single lexicographic ordering.
        order = torch.argsort(destination, stable=True)
        order = order[torch.argsort(source[order], stable=True)]
        return order[torch.argsort(edge_time[order], stable=True)]

    def update_memory(
        self,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        num_nodes: int | None = None,
    ) -> MemoryState:
        """Fold one batch of interactions into memory.

        Args:
            edge_index: ``[2, E]`` interactions.
            edge_time: ``[E]`` timestamps.
            edge_attr: ``[E, edge_dim]`` edge features, when ``edge_dim > 0``.
            num_nodes: Memory table size, if not already fixed.

        Returns:
            The updated :class:`MemoryState`.
        """
        edge_index = validate_temporal_batch(edge_index, edge_time, edge_attr, self.edge_dim)
        device = edge_index.device

        size = num_nodes if num_nodes is not None else self.num_nodes
        if size is None:
            size = int(edge_index.max()) + 1 if edge_index.numel() else 1
        state = self._ensure_memory(size, device)

        if edge_index.numel() == 0:
            return state

        order = self._canonical_order(edge_index, edge_time)
        source = edge_index[0][order]
        destination = edge_index[1][order]
        times = edge_time[order].float()
        features = (
            edge_attr[order]
            if edge_attr is not None and self.edge_dim > 0
            else torch.zeros(source.size(0), 0, device=device)
        )

        # Elapsed time since each endpoint was last touched. Clamped at zero
        # so an out-of-order event cannot produce a negative "time since",
        # which the encoder has no meaning for.
        elapsed = (times - state.last_update[destination]).clamp(min=0.0)
        time_encoding = self.time_encoder(elapsed)

        messages = self.message_function(
            state.memory[source], state.memory[destination], time_encoding, features
        )
        aggregated = aggregate_messages(
            messages, destination, state.memory.size(0), self.message_aggregation
        )

        # Only nodes that actually received a message are updated; running the
        # GRU over untouched rows would drift memory for nodes that did
        # nothing this window.
        touched = torch.zeros(state.memory.size(0), dtype=torch.bool, device=device)
        touched[destination] = True
        touched[source] = True

        updated = self.memory_updater(aggregated, state.memory)
        memory = torch.where(touched.unsqueeze(-1), updated, state.memory)

        last_update = state.last_update.clone()
        last_update = last_update.index_put((destination,), times, accumulate=False)
        last_update = last_update.index_put((source,), times, accumulate=False)

        self._state = MemoryState(memory=memory, last_update=last_update)
        return self._state

    # -- the standard interface -----------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor | None = None,
        node_time: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score nodes, updating memory from this batch first.

        Signature and return match the other temporal models in this package
        (``TemporalGCN`` and friends), so the same training loop drives all of
        them.

        Args:
            x: ``[N, input_dim]`` node features.
            edge_index: ``[2, E]`` interactions in this snapshot.
            edge_time: ``[E]`` interaction timestamps. Zeros when omitted.
            node_time: Accepted for interface compatibility; TGN derives its
                timing from ``edge_time`` and the memory's own last-update
                stamps.
            edge_attr: ``[E, edge_dim]`` edge features.

        Returns:
            ``[N, output_dim]`` log-probabilities.
        """
        if x.dim() != 2:
            raise ValueError(f"x must have shape [N, F], got {tuple(x.shape)}")
        if x.size(1) != self.input_dim:
            raise ValueError(f"x has {x.size(1)} features but the model expects {self.input_dim}")

        del node_time  # accepted for interface parity; see the docstring

        if edge_time is None:
            edge_time = torch.zeros(edge_index.size(1), device=x.device)

        state = self.update_memory(edge_index, edge_time, edge_attr, num_nodes=x.size(0))

        memory = state.memory[: x.size(0)]
        return F.log_softmax(self.embedding(torch.cat([x, memory], dim=-1)), dim=1)

    def forward_stream(
        self,
        snapshots: Sequence[dict[str, Any]],
        reset: bool = True,
        detach_between: bool = True,
    ) -> list[torch.Tensor]:
        """Run a sequence of snapshots in order, carrying memory across.

        Args:
            snapshots: Dicts with ``x`` and ``edge_index`` and optionally
                ``edge_time`` and ``edge_attr`` — the shape
                :func:`~astroml.features.graph.snapshot.iter_db_snapshots`
                output converts to.
            reset: Clear memory before the first snapshot. The default,
                because a stream should not inherit the previous stream's
                state.
            detach_between: Detach memory between snapshots so the autograd
                graph does not grow with the length of the stream.

        Returns:
            One ``[N, output_dim]`` prediction tensor per snapshot.
        """
        if reset:
            first = snapshots[0]["x"] if snapshots else None
            self.reset_memory(
                (
                    self.num_nodes
                    if self.num_nodes is not None
                    else (first.size(0) if first is not None else 1)
                ),
                device=first.device if first is not None else None,
            )

        outputs: list[torch.Tensor] = []
        for snapshot in snapshots:
            outputs.append(
                self.forward(
                    snapshot["x"],
                    snapshot["edge_index"],
                    edge_time=snapshot.get("edge_time"),
                    edge_attr=snapshot.get("edge_attr"),
                )
            )
            if detach_between:
                self.detach_memory()
        return outputs


def validate_temporal_batch(
    edge_index: torch.Tensor,
    edge_time: torch.Tensor,
    edge_attr: torch.Tensor | None,
    edge_dim: int,
) -> torch.Tensor:
    """Check a temporal batch is internally consistent.

    Length mismatches between ``edge_index`` and ``edge_time`` are the classic
    way to end up silently training on the wrong timestamps, so they are
    refused here rather than broadcast away.
    """
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"edge_index must be a Tensor, got {type(edge_index).__name__}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    edge_index = edge_index.long()
    num_edges = edge_index.size(1)

    if edge_time.dim() != 1 or edge_time.size(0) != num_edges:
        raise ValueError(f"edge_time must have shape [{num_edges}], got {tuple(edge_time.shape)}")
    if edge_index.numel() and int(edge_index.min()) < 0:
        raise ValueError("edge_index must not contain negative node indices")

    if edge_dim > 0:
        if edge_attr is None:
            raise ValueError(f"edge_dim is {edge_dim} but no edge_attr was given")
        if edge_attr.dim() != 2 or edge_attr.size(0) != num_edges:
            raise ValueError(
                f"edge_attr must have shape [{num_edges}, {edge_dim}], "
                f"got {tuple(edge_attr.shape)}"
            )
        if edge_attr.size(1) != edge_dim:
            raise ValueError(
                f"edge_attr has {edge_attr.size(1)} features but the model expects {edge_dim}"
            )

    return edge_index


__all__ = [
    "MESSAGE_AGGREGATIONS",
    "MemoryState",
    "MemoryUpdater",
    "MessageFunction",
    "TemporalGraphNetwork",
    "TimeEncoder",
    "aggregate_messages",
    "validate_temporal_batch",
]
