"""v1 training metrics: AUC with one class, precision/recall at a threshold, non-finite loss.

Covers issues #1209, #1210 and #1212. Per-epoch train and validation loss (#1207)
is covered with the training loop in ``test_train_gcn_loop.py``.
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from astroml.tracking.training_report import TrainingReport  # noqa: E402
from astroml.training.graph_objectives import (  # noqa: E402
    LinkPredictionObjective,
    NonFiniteLossError,
    ensure_finite_loss,
)
from astroml.utils.exceptions import AstroMLError, ModelError  # noqa: E402


def _t(values: list[float]) -> "torch.Tensor":
    return torch.tensor(values, dtype=torch.float32)


class TestAucWithOneClass:
    """#1209: an undefined AUC must be distinguishable from a real 0.5."""

    def test_only_positive_edges_is_flagged_undefined(self) -> None:
        metrics = LinkPredictionObjective().metrics(_t([2.0, 1.0, 3.0]), _t([1.0, 1.0, 1.0]))

        assert metrics["auc"] == 0.5
        assert metrics["auc_defined"] == 0.0

    def test_only_negative_edges_is_flagged_undefined(self) -> None:
        metrics = LinkPredictionObjective().metrics(_t([2.0, -1.0]), _t([0.0, 0.0]))

        assert metrics["auc"] == 0.5
        assert metrics["auc_defined"] == 0.0

    def test_both_classes_present_is_flagged_defined(self) -> None:
        metrics = LinkPredictionObjective().metrics(_t([2.0, -2.0]), _t([1.0, 0.0]))

        assert metrics["auc"] == 1.0
        assert metrics["auc_defined"] == 1.0

    def test_a_defined_auc_of_exactly_one_half_is_not_flagged(self) -> None:
        # Same scores for both classes: a real coin-flip model, not a missing value.
        metrics = LinkPredictionObjective().metrics(_t([1.0, 1.0]), _t([1.0, 0.0]))

        assert metrics["auc"] == pytest.approx(0.5)
        assert metrics["auc_defined"] == 1.0

    def test_no_edges_at_all_is_flagged_undefined(self) -> None:
        metrics = LinkPredictionObjective().metrics(_t([]), _t([]))

        assert metrics["auc"] == 0.5
        assert metrics["auc_defined"] == 0.0

    def test_the_single_class_value_is_unchanged_for_existing_callers(self) -> None:
        # The 0.5 itself is part of the documented contract and stays.
        objective = LinkPredictionObjective()
        assert objective.metrics(_t([1.0, 2.0]), torch.ones(2))["auc"] == 0.5
        assert objective.metrics(_t([1.0, 2.0]), torch.zeros(2))["auc"] == 0.5


class TestBestEpochIgnoresUndefinedMonitor:
    """#1209: a degenerate validation split can never be picked as the best epoch."""

    @staticmethod
    def _report() -> TrainingReport:
        return TrainingReport(LinkPredictionObjective())

    def test_an_undefined_auc_does_not_beat_a_real_but_lower_one(self) -> None:
        report = self._report()
        # Epoch 1: a genuinely weak model, AUC 0.4. Epoch 2: split had one class,
        # so AUC is reported as 0.5 but means nothing.
        report.record(1, val={"auc": 0.4, "auc_defined": 1.0})
        report.record(2, val={"auc": 0.5, "auc_defined": 0.0})

        assert report.best is not None
        assert report.best.epoch == 1

    def test_there_is_no_best_epoch_when_every_epoch_is_undefined(self) -> None:
        report = self._report()
        report.record(1, val={"auc": 0.5, "auc_defined": 0.0})
        report.record(2, val={"auc": 0.5, "auc_defined": 0.0})

        assert report.best is None
        assert report.to_dict()["best_epoch"] is None

    def test_an_undefined_validation_value_does_not_fall_back_to_training(self) -> None:
        report = self._report()
        report.record(1, train={"auc": 0.99}, val={"auc": 0.5, "auc_defined": 0.0})

        assert report.best is None

    def test_a_metric_without_a_defined_flag_is_treated_as_defined(self) -> None:
        # Older records and other objectives carry no ``*_defined`` keys.
        report = self._report()
        report.record(1, val={"auc": 0.7})
        report.record(2, val={"auc": 0.6})

        assert report.best is not None
        assert report.best.epoch == 1

    def test_the_objective_and_report_work_together_end_to_end(self) -> None:
        objective = LinkPredictionObjective()
        report = TrainingReport(objective)

        weak = objective.evaluate(_t([-1.0, 1.0]), _t([1.0, 0.0]))  # AUC 0.0, defined
        one_class = objective.evaluate(_t([1.0, 2.0]), _t([1.0, 1.0]))  # AUC undefined
        report.record(1, val=weak)
        report.record(2, val=one_class)

        assert weak["auc"] == 0.0 and one_class["auc"] == 0.5
        assert report.best is not None and report.best.epoch == 1


class TestPrecisionAndRecall:
    """#1210: precision and recall at a documented threshold."""

    def test_known_values_at_the_default_threshold(self) -> None:
        # Predicted positive: scores 3, 1 (probability > 0.5). Actual positives: first and third.
        scores = _t([3.0, 1.0, -1.0, -3.0])
        labels = _t([1.0, 0.0, 1.0, 0.0])

        metrics = LinkPredictionObjective().metrics(scores, labels)

        assert metrics["precision"] == pytest.approx(0.5)  # 1 of 2 predicted
        assert metrics["recall"] == pytest.approx(0.5)  # 1 of 2 actual
        assert metrics["accuracy"] == pytest.approx(0.5)

    def test_a_score_exactly_at_the_threshold_is_not_predicted(self) -> None:
        # The rule is strictly greater than the threshold: logit 0 is probability 0.5.
        metrics = LinkPredictionObjective().metrics(_t([0.0, 0.0]), _t([1.0, 0.0]))

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_a_higher_threshold_trades_recall_for_precision(self) -> None:
        # Probabilities: 0.95, 0.7, 0.4, 0.1 with labels 1, 0, 1, 0.
        scores = torch.logit(_t([0.95, 0.7, 0.4, 0.1]))
        labels = _t([1.0, 0.0, 1.0, 0.0])

        loose = LinkPredictionObjective(threshold=0.3).metrics(scores, labels)
        strict = LinkPredictionObjective(threshold=0.8).metrics(scores, labels)

        assert (loose["precision"], loose["recall"]) == (pytest.approx(2 / 3), 1.0)
        assert (strict["precision"], strict["recall"]) == (1.0, pytest.approx(0.5))

    def test_the_threshold_does_not_change_ranking_metrics(self) -> None:
        scores = torch.logit(_t([0.95, 0.7, 0.4, 0.1]))
        labels = _t([1.0, 0.0, 1.0, 0.0])

        low = LinkPredictionObjective(threshold=0.2).metrics(scores, labels)
        high = LinkPredictionObjective(threshold=0.9).metrics(scores, labels)

        assert low["auc"] == high["auc"]
        assert low["average_precision"] == high["average_precision"]

    def test_accuracy_uses_the_same_threshold(self) -> None:
        scores = torch.logit(_t([0.95, 0.7, 0.4, 0.1]))
        labels = _t([1.0, 0.0, 1.0, 0.0])

        assert LinkPredictionObjective(threshold=0.5).metrics(scores, labels)["accuracy"] == 0.5
        assert LinkPredictionObjective(threshold=0.8).metrics(scores, labels)["accuracy"] == 0.75

    def test_precision_is_zero_when_nothing_is_predicted_positive(self) -> None:
        metrics = LinkPredictionObjective().metrics(_t([-2.0, -3.0]), _t([1.0, 0.0]))

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_recall_is_zero_when_there_are_no_positive_edges(self) -> None:
        metrics = LinkPredictionObjective().metrics(_t([2.0, -2.0]), _t([0.0, 0.0]))

        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0  # one predicted positive, none correct

    def test_perfect_predictions_score_one(self) -> None:
        metrics = LinkPredictionObjective().metrics(_t([4.0, 4.0, -4.0]), _t([1.0, 1.0, 0.0]))

        assert (metrics["precision"], metrics["recall"], metrics["accuracy"]) == (1.0, 1.0, 1.0)

    def test_empty_input_reports_zeros(self) -> None:
        metrics = LinkPredictionObjective().metrics(_t([]), _t([]))

        assert metrics["precision"] == 0.0 and metrics["recall"] == 0.0

    def test_the_default_matches_the_previous_accuracy_rule(self) -> None:
        # Before thresholds existed, accuracy compared ``score > 0`` with the label.
        generator = torch.Generator().manual_seed(3)
        scores = torch.randn(200, generator=generator)
        labels = (torch.rand(200, generator=generator) > 0.5).float()

        previous = float(((scores > 0).double() == labels.double()).double().mean())

        assert LinkPredictionObjective().metrics(scores, labels)["accuracy"] == previous

    def test_agrees_with_scikit_learn(self) -> None:
        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        generator = torch.Generator().manual_seed(11)
        scores = 2 * torch.randn(500, generator=generator)
        labels = (torch.rand(500, generator=generator) > 0.6).float()

        for threshold in (0.3, 0.5, 0.75):
            metrics = LinkPredictionObjective(threshold=threshold).metrics(scores, labels)
            predicted = (torch.sigmoid(scores) > threshold).numpy()
            actual = labels.numpy()

            assert metrics["precision"] == pytest.approx(
                sklearn_metrics.precision_score(actual, predicted, zero_division=0)
            )
            assert metrics["recall"] == pytest.approx(
                sklearn_metrics.recall_score(actual, predicted, zero_division=0)
            )

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5, float("nan")])
    def test_a_threshold_outside_the_open_unit_interval_is_rejected(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            LinkPredictionObjective(threshold=threshold)

    def test_the_threshold_is_exposed(self) -> None:
        assert LinkPredictionObjective().threshold == 0.5
        assert LinkPredictionObjective(threshold=0.8).threshold == 0.8


class TestEnsureFiniteLoss:
    """#1212: a non-finite loss is a training failure."""

    def test_a_finite_value_is_returned_as_a_float(self) -> None:
        value = ensure_finite_loss(torch.tensor(0.25), epoch=3)

        assert value == 0.25 and isinstance(value, float)

    def test_a_plain_number_is_accepted(self) -> None:
        assert ensure_finite_loss(1, epoch=1) == 1.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_raise(self, bad: float) -> None:
        with pytest.raises(NonFiniteLossError):
            ensure_finite_loss(torch.tensor(bad), epoch=7)

    def test_the_error_names_the_epoch_phase_and_value(self) -> None:
        with pytest.raises(NonFiniteLossError) as raised:
            ensure_finite_loss(float("nan"), epoch=7, phase="val")

        error = raised.value
        assert "val" in str(error) and "epoch 7" in str(error)
        assert error.context["epoch"] == 7
        assert error.context["phase"] == "val"
        assert math.isnan(error.context["value"])

    def test_it_is_a_model_error_with_a_stable_code(self) -> None:
        with pytest.raises(NonFiniteLossError) as raised:
            ensure_finite_loss(float("inf"), epoch=1)

        error = raised.value
        assert isinstance(error, ModelError) and isinstance(error, AstroMLError)
        assert error.code == "NON_FINITE_LOSS"
        assert error.retryable is False

    def test_the_public_form_is_json_safe_and_generic(self) -> None:
        with pytest.raises(NonFiniteLossError) as raised:
            ensure_finite_loss(float("nan"), epoch=12, phase="train")

        public = json.dumps(raised.value.to_public_dict())
        assert "NON_FINITE_LOSS" in public
        assert "12" not in public

    def test_a_diverged_link_prediction_loss_is_caught(self) -> None:
        objective = LinkPredictionObjective()
        loss = objective.loss(_t([float("nan"), 1.0]), _t([1.0, 0.0]))

        with pytest.raises(NonFiniteLossError):
            ensure_finite_loss(loss, epoch=2)
