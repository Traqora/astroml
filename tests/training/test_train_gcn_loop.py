"""The GCN training loop: per-epoch loss (#1207) and non-finite loss as a failure (#1212).

``train()`` normally downloads Cora and starts a metrics server. These tests swap in
a small in-memory dataset and a no-op server, so they run offline on a CPU in a
couple of seconds and exercise the real loop.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
train_gcn = pytest.importorskip("astroml.training.train_gcn")

from astroml.tracking.training_report import TrainingReport  # noqa: E402
from astroml.training.graph_objectives import (  # noqa: E402
    NodeClassificationObjective,
    NonFiniteLossError,
)

NUM_NODES = 30
NUM_CLASSES = 3


class _Data(SimpleNamespace):
    def to(self, _device: Any) -> "_Data":
        return self


class _Dataset:
    num_node_features = 8
    num_classes = NUM_CLASSES

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(5)
        labels = torch.arange(NUM_NODES) % NUM_CLASSES
        x = 0.3 * torch.randn(NUM_NODES, 8, generator=generator)
        x[torch.arange(NUM_NODES), labels] += 1.0
        sources = torch.arange(NUM_NODES)
        edge_index = torch.stack([sources, (sources + NUM_CLASSES) % NUM_NODES])

        order = torch.randperm(NUM_NODES, generator=generator)
        masks = {
            name: torch.zeros(NUM_NODES, dtype=torch.bool) for name in ("train", "val", "test")
        }
        masks["train"][order[:15]] = True
        masks["val"][order[15:24]] = True
        masks["test"][order[24:]] = True
        self._data = _Data(
            x=x,
            edge_index=edge_index,
            y=labels,
            num_nodes=NUM_NODES,
            train_mask=masks["train"],
            val_mask=masks["val"],
            test_mask=masks["test"],
        )

    def __getitem__(self, _index: int) -> _Data:
        return self._data


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_gcn, "Planetoid", lambda **_kwargs: _Dataset())
    monkeypatch.setattr(train_gcn, "start_metrics_server", lambda *args, **kwargs: None)
    torch.manual_seed(0)


class _DivergingObjective(NodeClassificationObjective):
    """Returns a NaN loss on one chosen call to :meth:`loss`.

    The loop calls ``loss`` twice per epoch, training first and then validation,
    so call ``2k - 1`` is epoch ``k``'s training loss and call ``2k`` its validation loss.
    """

    def __init__(self, nan_on_call: int) -> None:
        super().__init__(log_input=True)
        self.nan_on_call = nan_on_call
        self.calls = 0

    def loss(self, outputs: Any, targets: Any) -> Any:
        self.calls += 1
        value = super().loss(outputs, targets)
        return value * float("nan") if self.calls == self.nan_on_call else value


class TestLossRecordedEveryEpoch:
    """#1207: train and validation loss for every epoch, not every ``eval_every``."""

    def test_every_epoch_has_a_train_and_a_validation_loss(self) -> None:
        report = train_gcn.train(epochs=8, eval_every=100)

        assert [record.epoch for record in report.records] == list(range(1, 9))
        for record in report.records:
            assert set(record.train) == {"loss"}
            assert "loss" in record.val
            assert record.train["loss"] > 0 and record.val["loss"] > 0

    def test_the_full_metric_set_stays_on_the_evaluation_cadence(self) -> None:
        report = train_gcn.train(epochs=6, eval_every=3)

        by_epoch = {record.epoch: record for record in report.records}
        # Evaluation epochs carry the full set; the rest carry loss only.
        for epoch in (3, 6):
            assert {"loss", "accuracy", "macro_f1"} <= set(by_epoch[epoch].val)
        for epoch in (1, 2, 4, 5):
            assert set(by_epoch[epoch].val) == {"loss"}

    def test_the_last_epoch_is_always_fully_evaluated(self) -> None:
        report = train_gcn.train(epochs=5, eval_every=100)

        assert "accuracy" in report.latest.val

    def test_the_validation_loss_is_a_different_number_from_the_training_loss(self) -> None:
        report = train_gcn.train(epochs=4, eval_every=100)

        # Computed on different nodes (and with dropout off), so they should not match.
        assert any(record.train["loss"] != record.val["loss"] for record in report.records)

    def test_the_training_loss_goes_down(self) -> None:
        report = train_gcn.train(epochs=40, eval_every=100)

        first, last = report.records[0], report.records[-1]
        assert last.train["loss"] < first.train["loss"]

    def test_the_report_round_trips_through_json(self) -> None:
        report = train_gcn.train(epochs=6, eval_every=3)

        payload = report.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert all("loss" in entry["val"] for entry in payload["history"])

    def test_a_caller_supplied_report_receives_the_per_epoch_records(self) -> None:
        report = TrainingReport(NodeClassificationObjective(log_input=True))

        returned = train_gcn.train(report=report, epochs=3, eval_every=100)

        assert returned is report and len(report) == 3


class TestNonFiniteLoss:
    """#1212: a NaN or infinite loss stops the run with a clear error."""

    def test_a_nan_training_loss_raises_with_the_epoch(self) -> None:
        objective = _DivergingObjective(nan_on_call=5)  # epoch 3, training loss

        with pytest.raises(NonFiniteLossError) as raised:
            train_gcn.train(objective=objective, epochs=10, eval_every=100)

        assert raised.value.context["epoch"] == 3
        assert raised.value.context["phase"] == "train"
        assert raised.value.code == "NON_FINITE_LOSS"

    def test_a_nan_validation_loss_raises_with_the_epoch(self) -> None:
        objective = _DivergingObjective(nan_on_call=4)  # epoch 2, validation loss

        with pytest.raises(NonFiniteLossError) as raised:
            train_gcn.train(objective=objective, epochs=10, eval_every=100)

        assert raised.value.context["epoch"] == 2
        assert raised.value.context["phase"] == "val"

    def test_training_stops_before_the_optimiser_steps_on_a_non_finite_loss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        steps = []
        original_step = torch.optim.Adam.step

        def counting_step(self: Any, *args: Any, **kwargs: Any) -> Any:
            steps.append(1)
            return original_step(self, *args, **kwargs)

        monkeypatch.setattr(torch.optim.Adam, "step", counting_step)

        with pytest.raises(NonFiniteLossError):
            train_gcn.train(objective=_DivergingObjective(nan_on_call=5), epochs=10)  # epoch 3

        # Epochs 1 and 2 stepped; epoch 3 failed before backward() and step().
        assert len(steps) == 2

    def test_the_epochs_before_the_failure_are_kept_and_valid_json(self) -> None:
        report = TrainingReport(NodeClassificationObjective(log_input=True))

        with pytest.raises(NonFiniteLossError):
            train_gcn.train(
                objective=_DivergingObjective(nan_on_call=7),  # epoch 4, training loss
                report=report,
                epochs=10,
                eval_every=100,
            )

        assert [record.epoch for record in report.records] == [1, 2, 3]
        # No NaN reached the report, so it is still valid, strict JSON.
        json.dumps(report.to_dict(), allow_nan=False)

    def test_a_run_that_stays_finite_is_unaffected(self) -> None:
        report = train_gcn.train(epochs=5, eval_every=100)

        assert len(report) == 5
