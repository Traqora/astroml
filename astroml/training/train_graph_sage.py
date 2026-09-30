"""Training and evaluation wiring for :class:`~astroml.models.graph_sage.GraphSAGE` — issue #736.

:mod:`astroml.training.train_gcn` is a script: it downloads Planetoid, trains
for a fixed 200 epochs and prints. That is fine as a demo but impossible to
call from a pipeline or a test. This module keeps the same training recipe
but exposes it as functions over tensors you already hold, so the same code
path serves the CLI, the benchmark harness and the unit tests.

Runs are reproducible: :func:`train_graph_sage` seeds torch from the config
before the model is constructed, so two runs with the same config and the
same data produce the same weights.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

from astroml.models.graph_sage import AGGREGATIONS, GraphSAGE, validate_edge_index


@dataclass
class GraphSAGEConfig:
    """Hyper-parameters for a GraphSAGE training run."""

    hidden_dim: int = 64
    hidden_dims: Sequence[int] | None = None
    num_layers: int = 2
    aggregator: str = "mean"
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    # Stop once validation accuracy has not improved for this many epochs.
    # None trains the full schedule.
    early_stopping_patience: int | None = 20
    seed: int = 42
    device: str = "cpu"
    # Prometheus counters are opt-in: importing the metrics module registers
    # collectors process-wide, which is unwanted inside a test run.
    export_metrics: bool = False

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.aggregator not in AGGREGATIONS:
            raise ValueError(
                f"unknown aggregator {self.aggregator!r}; expected one of {AGGREGATIONS}"
            )
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1 or None, "
                f"got {self.early_stopping_patience}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["hidden_dims"] is not None:
            payload["hidden_dims"] = list(payload["hidden_dims"])
        return payload


@dataclass
class TrainingHistory:
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


def _check_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    masks: dict[str, torch.Tensor | None],
) -> None:
    if x.dim() != 2:
        raise ValueError(f"x must have shape [N, F], got {tuple(x.shape)}")
    num_nodes = x.size(0)
    validate_edge_index(edge_index, num_nodes)

    if y.dim() != 1 or y.size(0) != num_nodes:
        raise ValueError(f"y must have shape [{num_nodes}], got {tuple(y.shape)}")

    for name, mask in masks.items():
        if mask is None:
            continue
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a bool tensor, got {mask.dtype}")
        if mask.size(0) != num_nodes:
            raise ValueError(f"{name} must cover all {num_nodes} nodes, got {mask.size(0)}")
        if not bool(mask.any()):
            raise ValueError(f"{name} selects no nodes")


@torch.no_grad()
def evaluate(
    model: GraphSAGE,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Accuracy and NLL loss over ``mask`` (or the whole graph)."""
    was_training = model.training
    model.eval()
    try:
        logits = model(x, edge_index)
        if mask is None:
            mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        selected = logits[mask]
        targets = y[mask]
        loss = float(F.nll_loss(selected, targets))
        correct = int((selected.argmax(dim=1) == targets).sum())
        total = int(mask.sum())
        return {
            "loss": loss,
            "accuracy": correct / total if total else 0.0,
            "num_nodes": float(total),
        }
    finally:
        model.train(was_training)


def train_graph_sage(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None = None,
    config: GraphSAGEConfig | None = None,
    model: GraphSAGE | None = None,
) -> tuple[GraphSAGE, TrainingHistory]:
    """Train a GraphSAGE classifier on a single graph.

    Args:
        x: ``[N, F]`` node features.
        edge_index: ``[2, E]`` adjacency.
        y: ``[N]`` integer class labels.
        train_mask: ``[N]`` bool mask of the supervised nodes.
        val_mask: Optional ``[N]`` bool validation mask. Required for early
            stopping and for restoring the best weights.
        config: Hyper-parameters. Defaults to :class:`GraphSAGEConfig`.
        model: An existing model to continue training. Built from ``config``
            when omitted.

    Returns:
        ``(model, history)``. When ``val_mask`` is given, the returned model
        carries the weights from the best validation epoch, not the last one
        — training past the optimum otherwise silently hands back a worse
        model than the run actually found.
    """
    config = config or GraphSAGEConfig()
    config.validate()
    _check_inputs(x, edge_index, y, {"train_mask": train_mask, "val_mask": val_mask})

    # Seed before constructing the model so the initial weights — not just
    # the dropout draws — are part of what the seed pins down.
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    x = x.to(device)
    edge_index = validate_edge_index(edge_index, x.size(0)).to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    if val_mask is not None:
        val_mask = val_mask.to(device)

    num_classes = int(y.max()) + 1
    if model is None:
        model = GraphSAGE(
            input_dim=x.size(1),
            hidden_dim=config.hidden_dim,
            output_dim=num_classes,
            dropout=config.dropout,
            num_layers=config.num_layers,
            aggregator=config.aggregator,
            hidden_dims=config.hidden_dims,
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    metrics = _load_metrics() if config.export_metrics else None
    if metrics is not None:
        metrics["MODEL_PARAMETERS"].labels(model_type="graph_sage").set(
            sum(p.numel() for p in model.parameters())
        )
        metrics["LEARNING_RATE"].labels(model_type="graph_sage").set(config.learning_rate)

    history = TrainingHistory()
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.nll_loss(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        history.train_loss.append(float(loss.detach()))
        history.epochs_run = epoch

        if metrics is not None:
            metrics["TRAINING_EPOCHS_TOTAL"].labels(
                model_type="graph_sage", dataset="stellar"
            ).inc()
            metrics["TRAINING_LOSS"].labels(
                model_type="graph_sage", dataset="stellar", phase="train"
            ).set(float(loss.detach()))

        if val_mask is None:
            continue

        val = evaluate(model, x, edge_index, y, val_mask)
        history.val_loss.append(val["loss"])
        history.val_accuracy.append(val["accuracy"])

        if metrics is not None:
            metrics["TRAINING_ACCURACY"].labels(
                model_type="graph_sage", dataset="stellar", phase="val"
            ).set(val["accuracy"])

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


def _load_metrics() -> dict[str, Any] | None:
    """Import the Prometheus collectors, or None when unavailable."""
    try:
        from astroml.training import metrics as training_metrics
    except ImportError:  # pragma: no cover - depends on optional prometheus_client
        return None
    return {
        name: getattr(training_metrics, name)
        for name in (
            "LEARNING_RATE",
            "MODEL_PARAMETERS",
            "TRAINING_ACCURACY",
            "TRAINING_EPOCHS_TOTAL",
            "TRAINING_LOSS",
        )
    }


__all__ = ["GraphSAGEConfig", "TrainingHistory", "evaluate", "train_graph_sage"]
