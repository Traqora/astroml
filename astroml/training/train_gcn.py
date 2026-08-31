import time

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from astroml.models.gcn import GCN
from astroml.tracking.training_report import TrainingReport
from astroml.training.graph_objectives import (
    GraphObjective,
    NodeClassificationObjective,
    validate_edge_index,
    validate_masks,
)
from astroml.training.metrics import (
    LEARNING_RATE,
    MODEL_PARAMETERS,
    TRAINING_ACCURACY,
    TRAINING_DURATION,
    TRAINING_EPOCHS_TOTAL,
    TRAINING_LOSS,
)
from astroml.training.metrics_server import start_metrics_server


def train(
    objective: GraphObjective | None = None,
    report: TrainingReport | None = None,
    epochs: int = 200,
    eval_every: int = 20,
):
    """Train a GCN on Cora.

    Args:
        objective: Loss and metrics to train against (issue #738). Defaults to
            :class:`NodeClassificationObjective` with ``log_input=True``,
            because :class:`~astroml.models.gcn.GCN` emits log-probabilities.
            Pass another objective to train the same loop on a different task.
        report: Collects per-epoch metrics. Defaults to a fresh report; pass
            one in to persist it to the registry afterwards with
            :meth:`~astroml.tracking.training_report.TrainingReport.persist`.
        epochs: Number of epochs.
        eval_every: Validation cadence, in epochs.

    Returns:
        The :class:`TrainingReport`, so a caller can persist or inspect it.
    """
    # Start Prometheus metrics server
    start_metrics_server()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    # Validate before the first forward pass (#738). An out-of-range edge or a
    # train/val mask overlap otherwise surfaces either as an opaque kernel
    # error or, worse, as a validation curve measured on training nodes.
    validate_edge_index(data.edge_index, data.num_nodes)
    validate_masks(
        {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        data.num_nodes,
    )

    objective = objective or NodeClassificationObjective(log_input=True)
    report = report if report is not None else TrainingReport(objective)

    model = GCN(
        input_dim=dataset.num_node_features,
        hidden_dim=16,
        output_dim=dataset.num_classes,
        dropout=0.5,
    ).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    MODEL_PARAMETERS.labels(model_type="gcn").set(total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    LEARNING_RATE.labels(model_type="gcn").set(0.01)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = objective.loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Update training metrics
        TRAINING_EPOCHS_TOTAL.labels(model_type="gcn", dataset="cora").inc()
        TRAINING_LOSS.labels(model_type="gcn", dataset="cora", phase="train").set(loss.item())

        train_metrics = {"loss": float(loss)}
        val_metrics: dict[str, float] = {}

        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
            val_metrics = objective.evaluate(logits[data.val_mask], data.y[data.val_mask])
            TRAINING_ACCURACY.labels(model_type="gcn", dataset="cora", phase="val").set(
                val_metrics.get("accuracy", 0.0)
            )
            print(
                f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f}"
            )

        # Log epoch duration
        epoch_duration = time.time() - epoch_start
        TRAINING_DURATION.labels(model_type="gcn", dataset="cora").observe(epoch_duration)

        report.record(
            epoch=epoch,
            train=train_metrics,
            val=val_metrics,
            duration_seconds=epoch_duration,
        )

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    test_metrics = objective.evaluate(logits[data.test_mask], data.y[data.test_mask])
    test_acc = test_metrics.get("accuracy", 0.0)
    TRAINING_ACCURACY.labels(model_type="gcn", dataset="cora", phase="test").set(test_acc)
    print(f"Test Accuracy: {test_acc:.4f}")

    best = report.best
    if best is not None:
        print(f"Best epoch: {best.epoch} ({report.monitor}={best.val.get(report.monitor)})")

    return report


def _accuracy(model: GCN, data, mask) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    return float((pred[mask] == data.y[mask]).sum()) / float(mask.sum())


if __name__ == "__main__":
    train()
