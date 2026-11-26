"""CPU-only smoke test of the baseline GCN path (issue #1202).

Runs the whole v1 baseline path end to end on a small synthetic graph: validate
the input, train a few epochs with the node-classification objective, record a
report, save and reload a checkpoint. It needs no GPU, no dataset download, no
network, no metrics server and no database, so it can run on any CI worker and
catches a broken baseline before the slower integration tests do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from astroml.models.checkpoint import load_gcn_checkpoint, save_gcn_checkpoint  # noqa: E402
from astroml.models.gcn import GCN  # noqa: E402
from astroml.models.graph_input import prepare_graph_input  # noqa: E402
from astroml.tracking.training_report import TrainingReport  # noqa: E402
from astroml.training.graph_objectives import NodeClassificationObjective  # noqa: E402

NUM_CLASSES = 3
NUM_NODES = 45
ISOLATED = (42, 43, 44)
EPOCHS = 60


def _synthetic_graph() -> dict[str, "torch.Tensor"]:
    """A learnable graph: class signal in the features, edges between same-class nodes."""
    generator = torch.Generator().manual_seed(7)
    labels = torch.arange(NUM_NODES) % NUM_CLASSES

    features = 0.3 * torch.randn(NUM_NODES, 8, generator=generator)
    features[torch.arange(NUM_NODES), labels] += 1.0

    sources, targets = [], []
    for node in range(NUM_NODES - len(ISOLATED)):
        same_class = [
            n for n in range(NUM_NODES - len(ISOLATED)) if n != node and labels[n] == labels[node]
        ]
        for neighbour in same_class[:2]:
            sources += [node, neighbour]
            targets += [neighbour, node]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    order = torch.randperm(NUM_NODES, generator=generator)
    train_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    val_mask = torch.zeros(NUM_NODES, dtype=torch.bool)
    train_mask[order[:27]] = True
    val_mask[order[27:]] = True
    return {
        "x": features,
        "edge_index": edge_index,
        "y": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
    }


@pytest.fixture(scope="module")
def run() -> dict[str, object]:
    """Train once; the tests below read different parts of the same run."""
    torch.manual_seed(0)
    graph = _synthetic_graph()
    prepared = prepare_graph_input(graph["x"], graph["edge_index"])

    objective = NodeClassificationObjective(log_input=True)
    report = TrainingReport(objective)
    model = GCN(prepared.x.size(1), 16, NUM_CLASSES, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(prepared.x, prepared.edge_index)
        loss = objective.loss(out[graph["train_mask"]], graph["y"][graph["train_mask"]])
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

        val: dict[str, float] = {}
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(prepared.x, prepared.edge_index)
            val = objective.evaluate(logits[graph["val_mask"]], graph["y"][graph["val_mask"]])
        report.record(epoch, train={"loss": float(loss)}, val=val)

    model.eval()
    return {
        "model": model,
        "graph": graph,
        "prepared": prepared,
        "report": report,
        "losses": losses,
    }


def test_the_graph_has_isolated_nodes_and_is_accepted(run: dict[str, object]) -> None:
    prepared = run["prepared"]
    assert prepared.num_isolated == len(ISOLATED)  # type: ignore[attr-defined]
    assert prepared.num_nodes == NUM_NODES  # type: ignore[attr-defined]


def test_everything_runs_on_the_cpu(run: dict[str, object]) -> None:
    model = run["model"]
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())  # type: ignore[attr-defined]
    prepared = run["prepared"]
    assert prepared.x.device.type == "cpu"  # type: ignore[attr-defined]


def test_the_training_loss_goes_down(run: dict[str, object]) -> None:
    losses = run["losses"]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < 0.5 * losses[0]  # type: ignore[index]


def test_the_trained_model_beats_chance_on_held_out_nodes(run: dict[str, object]) -> None:
    report = run["report"]
    accuracy = report.latest.val["accuracy"]  # type: ignore[attr-defined]
    assert accuracy > 0.8, f"validation accuracy {accuracy:.2f} is not above chance (0.33)"


def test_output_is_finite_for_isolated_nodes(run: dict[str, object]) -> None:
    prepared = run["prepared"]
    with torch.no_grad():
        out = run["model"](prepared.x, prepared.edge_index)  # type: ignore[operator, attr-defined]
    assert out.shape == (NUM_NODES, NUM_CLASSES)
    assert bool(torch.isfinite(out[list(ISOLATED)]).all())


def test_the_report_covers_every_epoch_and_is_json_serialisable(run: dict[str, object]) -> None:
    report = run["report"]
    payload = report.to_dict()  # type: ignore[attr-defined]

    assert payload["epochs_recorded"] == EPOCHS
    assert payload["best_epoch"] is not None
    assert payload["environment"]["packages"]["torch"] == torch.__version__
    assert json.loads(json.dumps(payload)) == payload


def test_a_saved_checkpoint_reproduces_the_predictions(
    run: dict[str, object], tmp_path: Path
) -> None:
    model, prepared = run["model"], run["prepared"]
    path = save_gcn_checkpoint(model, tmp_path / "baseline.pt")  # type: ignore[arg-type]

    reloaded = load_gcn_checkpoint(path).model
    with torch.no_grad():
        expected = model(prepared.x, prepared.edge_index)  # type: ignore[operator, attr-defined]
        actual = reloaded(prepared.x, prepared.edge_index)  # type: ignore[attr-defined]
    assert torch.allclose(expected, actual)
