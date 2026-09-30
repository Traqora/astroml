"""Training and evaluation wiring for the TGN — issue #737.

A temporal model needs a different loop from a static one. Snapshots must be
replayed in chronological order because memory carries between them, memory
has to be cleared at the start of every epoch, and evaluation must not leave
the model's state changed — an eval pass that silently advances memory makes
the next training epoch start somewhere the config never described.

This module handles all three, over a plain list of snapshot dicts, so the
same code path serves a pipeline, the benchmark harness and the smoke test
the issue asks for.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

from astroml.models.tgn import MESSAGE_AGGREGATIONS, TemporalGraphNetwork

#: One snapshot: ``x``, ``edge_index``, ``y`` and optionally ``edge_time``,
#: ``edge_attr`` and ``mask``.
Snapshot = dict[str, Any]


@dataclass
class TGNConfig:
    """Hyper-parameters for a TGN training run."""

    hidden_dim: int = 64
    memory_dim: int = 64
    time_dim: int = 32
    message_dim: int = 64
    edge_dim: int = 0
    dropout: float = 0.1
    message_aggregation: str = "mean"
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 50
    early_stopping_patience: int | None = 10
    # Clip the gradient: a recurrent memory update is the usual place for an
    # exploding gradient to appear, and it appears as a NaN loss several
    # epochs later rather than at the point of failure.
    grad_clip: float | None = 1.0
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.message_aggregation not in MESSAGE_AGGREGATIONS:
            raise ValueError(
                f"unknown message_aggregation {self.message_aggregation!r}; "
                f"expected one of {MESSAGE_AGGREGATIONS}"
            )
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1 or None, "
                f"got {self.early_stopping_patience}"
            )
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be > 0 or None, got {self.grad_clip}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TGNHistory:
    """What happened during a run."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = 0
    best_val_accuracy: float = 0.0
    stopped_early: bool = False
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_snapshots(snapshots: Sequence[Snapshot]) -> int:
    """Check a snapshot stream and return the node count it spans.

    Raises:
        ValueError: if the stream is empty, a snapshot is missing a required
            key, shapes disagree, or the snapshots are not in chronological
            order — replaying a temporal model out of order produces a
            confidently wrong result rather than an error.
    """
    if not snapshots:
        raise ValueError("snapshots must not be empty")

    num_nodes = 0
    previous_end: float | None = None

    for index, snapshot in enumerate(snapshots):
        for key in ("x", "edge_index", "y"):
            if key not in snapshot:
                raise ValueError(f"snapshot {index} is missing {key!r}")

        x, edge_index, y = snapshot["x"], snapshot["edge_index"], snapshot["y"]
        if x.dim() != 2:
            raise ValueError(f"snapshot {index}: x must have shape [N, F]")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"snapshot {index}: edge_index must have shape [2, E]")
        if y.dim() != 1 or y.size(0) != x.size(0):
            raise ValueError(f"snapshot {index}: y must have shape [{x.size(0)}]")

        mask = snapshot.get("mask")
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError(f"snapshot {index}: mask must be a bool tensor")
            if mask.size(0) != x.size(0):
                raise ValueError(f"snapshot {index}: mask must cover all {x.size(0)} nodes")

        edge_time = snapshot.get("edge_time")
        if edge_time is not None and edge_time.numel():
            start = float(edge_time.min())
            if previous_end is not None and start < previous_end:
                raise ValueError(
                    f"snapshot {index} starts at {start} but snapshot {index - 1} "
                    f"ran to {previous_end}; a temporal stream must be chronological"
                )
            previous_end = float(edge_time.max())

        num_nodes = max(num_nodes, x.size(0))
        if edge_index.numel():
            num_nodes = max(num_nodes, int(edge_index.max()) + 1)

    return num_nodes


def _snapshot_loss(
    model: TemporalGraphNetwork,
    snapshot: Snapshot,
) -> tuple[torch.Tensor, int, int]:
    """``(loss, correct, total)`` for one snapshot."""
    logits = model(
        snapshot["x"],
        snapshot["edge_index"],
        edge_time=snapshot.get("edge_time"),
        edge_attr=snapshot.get("edge_attr"),
    )
    mask = snapshot.get("mask")
    if mask is None:
        mask = torch.ones(snapshot["x"].size(0), dtype=torch.bool, device=logits.device)

    selected = logits[mask]
    targets = snapshot["y"][mask]
    if selected.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, 0, 0

    loss = F.nll_loss(selected, targets)
    correct = int((selected.argmax(dim=1) == targets).sum())
    return loss, correct, int(mask.sum())


@torch.no_grad()
def evaluate_tgn(
    model: TemporalGraphNetwork,
    snapshots: Sequence[Snapshot],
    restore_memory: bool = True,
) -> dict[str, float]:
    """Replay ``snapshots`` and report loss and accuracy.

    Args:
        restore_memory: Put the model's memory back as it was afterwards.
            On by default: evaluating is a read, and a read that advances the
            model's state would change the training run that follows it.
    """
    was_training = model.training
    model.eval()
    saved = model.get_memory().clone() if (restore_memory and model.memory_initialised) else None

    try:
        total_loss = 0.0
        correct = 0
        total = 0
        for snapshot in snapshots:
            loss, snapshot_correct, snapshot_total = _snapshot_loss(model, snapshot)
            total_loss += float(loss)
            correct += snapshot_correct
            total += snapshot_total

        return {
            "loss": total_loss / len(snapshots) if snapshots else 0.0,
            "accuracy": correct / total if total else 0.0,
            "num_nodes": float(total),
        }
    finally:
        if saved is not None:
            model._state = saved
        model.train(was_training)


def train_tgn(
    snapshots: Sequence[Snapshot],
    num_classes: int | None = None,
    val_snapshots: Sequence[Snapshot] | None = None,
    config: TGNConfig | None = None,
    model: TemporalGraphNetwork | None = None,
) -> tuple[TemporalGraphNetwork, TGNHistory]:
    """Train a TGN over a chronological snapshot stream.

    Args:
        snapshots: Training snapshots, oldest first.
        num_classes: Inferred from the labels when omitted.
        val_snapshots: Validation stream, replayed after each epoch. Required
            for early stopping and best-weight restore.
        config: Hyper-parameters. Defaults to :class:`TGNConfig`.
        model: An existing model to continue training.

    Returns:
        ``(model, history)``. With a validation stream the model carries the
        weights from the best epoch rather than the last.
    """
    config = config or TGNConfig()
    config.validate()

    num_nodes = validate_snapshots(snapshots)
    if val_snapshots:
        num_nodes = max(num_nodes, validate_snapshots(val_snapshots))

    # Seed before the model is built so the initial weights, not just the
    # dropout draws, are pinned by the seed.
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    if num_classes is None:
        num_classes = int(max(int(snapshot["y"].max()) for snapshot in snapshots)) + 1

    if model is None:
        model = TemporalGraphNetwork(
            input_dim=snapshots[0]["x"].size(1),
            hidden_dim=config.hidden_dim,
            output_dim=num_classes,
            memory_dim=config.memory_dim,
            time_dim=config.time_dim,
            edge_dim=config.edge_dim,
            message_dim=config.message_dim,
            num_nodes=num_nodes,
            dropout=config.dropout,
            message_aggregation=config.message_aggregation,
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history = TGNHistory()
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        # Every epoch replays the stream from a clean memory. Without this the
        # second epoch starts from the first one's final state and the run is
        # no longer the one the config describes.
        model.reset_memory(num_nodes, device=device)

        epoch_loss = 0.0
        for snapshot in snapshots:
            optimizer.zero_grad()
            loss, _, _ = _snapshot_loss(model, snapshot)
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            # Detach between snapshots: the memory tensor is carried forward,
            # and without this the graph grows with the length of the stream.
            model.detach_memory()
            epoch_loss += float(loss.detach())

        history.train_loss.append(epoch_loss / len(snapshots))
        history.epochs_run = epoch

        if not val_snapshots:
            continue

        val = evaluate_tgn(model, val_snapshots)
        history.val_loss.append(val["loss"])
        history.val_accuracy.append(val["accuracy"])

        if val["accuracy"] > history.best_val_accuracy or best_state is None:
            history.best_val_accuracy = val["accuracy"]
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                config.early_stopping_patience is not None
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                history.stopped_early = True
                break

    history.duration_seconds = time.perf_counter() - started

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


__all__ = [
    "Snapshot",
    "TGNConfig",
    "TGNHistory",
    "evaluate_tgn",
    "train_tgn",
    "validate_snapshots",
]
