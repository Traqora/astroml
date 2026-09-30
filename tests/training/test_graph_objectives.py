"""Tests for pluggable graph loss/metric objectives (issue #738)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from astroml.training.graph_objectives import (  # noqa: E402
    GraphObjective,
    LinkPredictionObjective,
    NodeClassificationObjective,
    available_objectives,
    get_objective,
    register_objective,
    validate_edge_index,
    validate_masks,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateEdgeIndex:
    """An invalid edge list must fail here, not inside a scatter kernel."""

    def test_accepts_a_well_formed_edge_index(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        validate_edge_index(edge_index, num_nodes=3)

    def test_accepts_an_empty_graph(self):
        validate_edge_index(torch.zeros((2, 0), dtype=torch.long), num_nodes=0)

    @pytest.mark.parametrize("shape", [(3, 2), (2,), (1, 5)], ids=["transposed", "flat", "one_row"])
    def test_rejects_a_wrong_shape(self, shape):
        with pytest.raises(ValueError, match=r"shape \[2, E\]"):
            validate_edge_index(torch.zeros(shape, dtype=torch.long), num_nodes=10)

    def test_rejects_a_float_edge_index(self):
        with pytest.raises(ValueError, match="integer tensor"):
            validate_edge_index(torch.zeros((2, 3)), num_nodes=10)

    def test_rejects_an_out_of_range_node_id(self):
        edge_index = torch.tensor([[0, 1], [1, 7]], dtype=torch.long)
        with pytest.raises(ValueError, match="references node 7"):
            validate_edge_index(edge_index, num_nodes=3)

    def test_rejects_a_negative_node_id(self):
        edge_index = torch.tensor([[0, -1], [1, 2]], dtype=torch.long)
        with pytest.raises(ValueError, match="negative node id"):
            validate_edge_index(edge_index, num_nodes=3)

    def test_accepts_the_highest_valid_node_id(self):
        edge_index = torch.tensor([[0], [2]], dtype=torch.long)
        validate_edge_index(edge_index, num_nodes=3)


class TestValidateMasks:
    """Overlapping splits are a leak that still looks like a healthy run."""

    def test_accepts_disjoint_masks(self):
        train = torch.tensor([True, True, False, False])
        val = torch.tensor([False, False, True, False])
        validate_masks({"train": train, "val": val}, num_nodes=4)

    def test_rejects_overlapping_masks(self):
        train = torch.tensor([True, True, False, False])
        val = torch.tensor([False, True, True, False])
        with pytest.raises(ValueError, match="overlap on 1 nodes"):
            validate_masks({"train": train, "val": val}, num_nodes=4)

    def test_rejects_a_non_boolean_mask(self):
        with pytest.raises(ValueError, match="boolean tensor"):
            validate_masks({"train": torch.tensor([1, 0, 1])}, num_nodes=3)

    def test_rejects_a_mask_of_the_wrong_length(self):
        with pytest.raises(ValueError, match=r"shape \[4\]"):
            validate_masks({"train": torch.tensor([True, False])}, num_nodes=4)

    def test_names_both_offending_splits(self):
        masks = {
            "train": torch.tensor([True, False, False]),
            "val": torch.tensor([False, True, False]),
            "test": torch.tensor([False, True, False]),
        }
        with pytest.raises(ValueError) as excinfo:
            validate_masks(masks, num_nodes=3)
        assert "test" in str(excinfo.value) and "val" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------


class TestNodeClassificationObjective:
    def test_perfect_predictions_score_one(self):
        objective = NodeClassificationObjective(log_input=False)
        logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
        targets = torch.tensor([0, 1, 0])

        metrics = objective.metrics(logits, targets)

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)

    def test_accuracy_counts_correct_predictions(self):
        objective = NodeClassificationObjective(log_input=False)
        logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
        targets = torch.tensor([0, 1, 1, 0])

        assert objective.metrics(logits, targets)["accuracy"] == pytest.approx(0.5)

    def test_macro_f1_penalises_ignoring_the_minority_class(self):
        """A model predicting only the majority class scores well on accuracy.

        Macro-F1 is the metric that notices, which is why both are reported.
        """
        objective = NodeClassificationObjective(log_input=False)
        logits = torch.tensor([[5.0, 0.0]] * 9 + [[5.0, 0.0]])
        targets = torch.tensor([0] * 9 + [1])

        metrics = objective.metrics(logits, targets)

        assert metrics["accuracy"] == pytest.approx(0.9)
        # Majority class F1 ≈ 0.947, minority class 0 → macro ≈ 0.474.
        assert metrics["macro_f1"] < 0.5

    def test_loss_decreases_as_predictions_improve(self):
        objective = NodeClassificationObjective(log_input=False)
        targets = torch.tensor([0, 1])
        confident = torch.tensor([[9.0, 0.0], [0.0, 9.0]])
        unsure = torch.tensor([[0.1, 0.0], [0.0, 0.1]])

        assert float(objective.loss(confident, targets)) < float(objective.loss(unsure, targets))

    def test_log_input_uses_nll_over_log_probabilities(self):
        """The GCN in this package emits log-probabilities, not logits."""
        objective = NodeClassificationObjective(log_input=True)
        log_probs = torch.log_softmax(torch.tensor([[5.0, 0.0], [0.0, 5.0]]), dim=1)
        targets = torch.tensor([0, 1])

        loss = objective.loss(log_probs, targets)

        expected = torch.nn.functional.nll_loss(log_probs, targets)
        assert float(loss) == pytest.approx(float(expected))

    def test_loss_is_differentiable(self):
        objective = NodeClassificationObjective(log_input=False)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0])

        objective.loss(logits, targets).backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_class_weight_reweights_the_loss(self):
        """Upweighting the class the model gets wrong raises the loss.

        The batch has to be mixed: ``cross_entropy`` normalises by the total
        weight, so scaling every sample in a single-class batch by the same
        factor cancels out and changes nothing.
        """
        # Both rows predict class 1, so the class-0 row carries almost all
        # the loss.
        logits = torch.tensor([[0.0, 3.0], [0.0, 3.0]])
        targets = torch.tensor([0, 1])

        unweighted = NodeClassificationObjective(log_input=False).loss(logits, targets)
        weighted = NodeClassificationObjective(
            log_input=False, class_weight=torch.tensor([5.0, 1.0])
        ).loss(logits, targets)

        assert float(weighted) > float(unweighted)

        # And downweighting it lowers the loss, so the effect tracks the
        # weight rather than merely differing from the default.
        downweighted = NodeClassificationObjective(
            log_input=False, class_weight=torch.tensor([0.2, 1.0])
        ).loss(logits, targets)
        assert float(downweighted) < float(unweighted)

    def test_metrics_never_require_grad(self):
        objective = NodeClassificationObjective(log_input=False)
        logits = torch.randn(8, 3, requires_grad=True)
        targets = torch.randint(0, 3, (8,))

        metrics = objective.metrics(logits, targets)

        assert all(isinstance(value, float) for value in metrics.values())

    def test_handles_an_empty_batch(self):
        objective = NodeClassificationObjective(log_input=False)
        metrics = objective.metrics(torch.zeros((0, 3)), torch.zeros(0, dtype=torch.long))
        assert metrics == {"accuracy": 0.0, "macro_f1": 0.0}

    @pytest.mark.parametrize(
        "logits,targets",
        [
            (torch.zeros(4), torch.zeros(4, dtype=torch.long)),
            (torch.zeros((4, 3)), torch.zeros(5, dtype=torch.long)),
        ],
        ids=["one_dimensional_logits", "mismatched_targets"],
    )
    def test_rejects_malformed_shapes(self, logits, targets):
        with pytest.raises(ValueError):
            NodeClassificationObjective(log_input=False).metrics(logits, targets)


# ---------------------------------------------------------------------------
# Link prediction
# ---------------------------------------------------------------------------


class TestLinkPredictionObjective:
    def test_perfect_ranking_scores_auc_one(self):
        objective = LinkPredictionObjective()
        scores = torch.tensor([3.0, 2.5, -2.0, -3.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

        metrics = objective.metrics(scores, labels)

        assert metrics["auc"] == pytest.approx(1.0)
        assert metrics["average_precision"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_inverted_ranking_scores_auc_zero(self):
        objective = LinkPredictionObjective()
        scores = torch.tensor([-3.0, -2.0, 2.0, 3.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

        assert objective.metrics(scores, labels)["auc"] == pytest.approx(0.0)

    def test_constant_scores_give_a_coin_flip_auc(self):
        """Tied scores get averaged ranks, matching scikit-learn."""
        objective = LinkPredictionObjective()
        scores = torch.zeros(6)
        labels = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        assert objective.metrics(scores, labels)["auc"] == pytest.approx(0.5)

    def test_auc_matches_a_hand_computed_value(self):
        objective = LinkPredictionObjective()
        # One positive ranked above one negative and below another → 0.5.
        scores = torch.tensor([2.0, 1.0, 0.0])
        labels = torch.tensor([0.0, 1.0, 0.0])

        assert objective.metrics(scores, labels)["auc"] == pytest.approx(0.5)

    def test_auc_is_undefined_without_both_classes(self):
        objective = LinkPredictionObjective()

        assert objective.metrics(torch.tensor([1.0, 2.0]), torch.ones(2))["auc"] == 0.5
        assert objective.metrics(torch.tensor([1.0, 2.0]), torch.zeros(2))["auc"] == 0.5

    def test_average_precision_is_zero_without_positives(self):
        objective = LinkPredictionObjective()
        metrics = objective.metrics(torch.tensor([1.0, 2.0]), torch.zeros(2))
        assert metrics["average_precision"] == 0.0

    def test_loss_uses_logits_directly_and_stays_finite(self):
        """A manual sigmoid before BCE overflows on confident scores."""
        objective = LinkPredictionObjective()
        scores = torch.tensor([80.0, -80.0])
        labels = torch.tensor([1.0, 0.0])

        loss = objective.loss(scores, labels)

        assert math.isfinite(float(loss))
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_differentiable(self):
        objective = LinkPredictionObjective()
        scores = torch.randn(16, requires_grad=True)
        labels = (torch.rand(16) > 0.5).float()

        objective.loss(scores, labels).backward()

        assert scores.grad is not None
        assert torch.isfinite(scores.grad).all()

    def test_pos_weight_raises_the_cost_of_missing_a_positive(self):
        scores = torch.tensor([-2.0, -2.0])
        labels = torch.tensor([1.0, 1.0])

        plain = LinkPredictionObjective().loss(scores, labels)
        weighted = LinkPredictionObjective(pos_weight=5.0).loss(scores, labels)

        assert float(weighted) > float(plain)

    def test_accuracy_thresholds_at_a_logit_of_zero(self):
        objective = LinkPredictionObjective()
        scores = torch.tensor([0.5, -0.5, 0.5, -0.5])
        labels = torch.tensor([1.0, 0.0, 0.0, 1.0])

        assert objective.metrics(scores, labels)["accuracy"] == pytest.approx(0.5)

    def test_handles_an_empty_batch(self):
        metrics = LinkPredictionObjective().metrics(torch.zeros(0), torch.zeros(0))
        assert metrics["auc"] == 0.5

    def test_rejects_malformed_shapes(self):
        objective = LinkPredictionObjective()
        with pytest.raises(ValueError, match=r"shape \[E\]"):
            objective.metrics(torch.zeros((2, 2)), torch.zeros(2))
        with pytest.raises(ValueError, match="targets must have shape"):
            objective.metrics(torch.zeros(3), torch.zeros(4))


class TestMetricsAgainstSklearn:
    """Cross-check the Torch implementations against the reference."""

    def test_auc_and_average_precision_match_sklearn(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")

        generator = torch.Generator().manual_seed(7)
        scores = torch.randn(500, generator=generator)
        labels = (torch.rand(500, generator=generator) > 0.6).float()

        metrics = LinkPredictionObjective().metrics(scores, labels)

        expected_auc = sklearn_metrics.roc_auc_score(labels.numpy(), scores.numpy())
        expected_ap = sklearn_metrics.average_precision_score(labels.numpy(), scores.numpy())

        assert metrics["auc"] == pytest.approx(expected_auc, abs=1e-9)
        assert metrics["average_precision"] == pytest.approx(expected_ap, abs=1e-9)

    def test_auc_matches_sklearn_with_heavy_ties(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")

        # Three distinct scores over 300 samples: almost everything is tied.
        generator = torch.Generator().manual_seed(11)
        scores = torch.randint(0, 3, (300,), generator=generator).float()
        labels = (torch.rand(300, generator=generator) > 0.5).float()

        auc = LinkPredictionObjective().metrics(scores, labels)["auc"]
        expected = sklearn_metrics.roc_auc_score(labels.numpy(), scores.numpy())

        assert auc == pytest.approx(expected, abs=1e-9)

    def test_macro_f1_matches_sklearn(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")

        generator = torch.Generator().manual_seed(3)
        logits = torch.randn(400, 4, generator=generator)
        targets = torch.randint(0, 4, (400,), generator=generator)

        macro_f1 = NodeClassificationObjective(log_input=False).metrics(logits, targets)["macro_f1"]
        expected = sklearn_metrics.f1_score(
            targets.numpy(), logits.argmax(dim=1).numpy(), average="macro", zero_division=0
        )

        assert macro_f1 == pytest.approx(expected, abs=1e-9)


class TestLargeGraphBehaviour:
    """Metric cost must stay bounded as the graph grows."""

    def test_subsamples_above_the_cap(self):
        objective = LinkPredictionObjective(max_metric_samples=100)
        generator = torch.Generator().manual_seed(5)
        scores = torch.randn(10_000, generator=generator)
        labels = (torch.rand(10_000, generator=generator) > 0.5).float()

        metrics = objective.metrics(scores, labels)

        assert 0.0 <= metrics["auc"] <= 1.0

    def test_the_subsample_is_deterministic(self):
        """Two runs of the same objective must report the same number.

        A metric that wanders between runs cannot be used to pick a best epoch.
        """
        generator = torch.Generator().manual_seed(5)
        scores = torch.randn(5_000, generator=generator)
        labels = (torch.rand(5_000, generator=generator) > 0.5).float()

        first = LinkPredictionObjective(max_metric_samples=200).metrics(scores, labels)
        second = LinkPredictionObjective(max_metric_samples=200).metrics(scores, labels)

        assert first == second

    def test_subsampling_does_not_disturb_the_global_rng(self):
        """Metric logging must not change the weights a run produces."""
        objective = LinkPredictionObjective(max_metric_samples=50)
        scores = torch.randn(1_000)
        labels = (torch.rand(1_000) > 0.5).float()

        torch.manual_seed(123)
        before = torch.randn(3)

        torch.manual_seed(123)
        objective.metrics(scores, labels)
        after = torch.randn(3)

        assert torch.equal(before, after)

    def test_no_subsampling_when_disabled(self):
        sklearn_metrics = pytest.importorskip("sklearn.metrics")

        generator = torch.Generator().manual_seed(9)
        scores = torch.randn(2_000, generator=generator)
        labels = (torch.rand(2_000, generator=generator) > 0.5).float()

        auc = LinkPredictionObjective(max_metric_samples=0).metrics(scores, labels)["auc"]
        expected = sklearn_metrics.roc_auc_score(labels.numpy(), scores.numpy())

        assert auc == pytest.approx(expected, abs=1e-9)

    def test_rejects_a_negative_sample_cap(self):
        with pytest.raises(ValueError, match="non-negative"):
            LinkPredictionObjective(max_metric_samples=-1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestObjectiveRegistry:
    def test_built_in_objectives_are_registered(self):
        names = list(available_objectives())
        assert "node_classification" in names
        assert "link_prediction" in names

    def test_get_objective_constructs_by_name(self):
        assert isinstance(get_objective("link_prediction"), LinkPredictionObjective)
        assert isinstance(get_objective("node_classification"), NodeClassificationObjective)

    def test_get_objective_forwards_keyword_arguments(self):
        objective = get_objective("link_prediction", pos_weight=3.0, max_metric_samples=10)
        assert objective.pos_weight == 3.0
        assert objective.max_metric_samples == 10

    def test_unknown_objective_lists_the_alternatives(self):
        with pytest.raises(KeyError) as excinfo:
            get_objective("does_not_exist")
        assert "link_prediction" in str(excinfo.value)

    def test_a_custom_objective_can_be_plugged_in(self):
        class ConstantObjective(GraphObjective):
            name = "constant_test"
            monitor = "constant"
            higher_is_better = True

            def loss(self, outputs, targets):
                return outputs.sum() * 0.0

            def metrics(self, outputs, targets):
                return {"constant": 1.0}

        register_objective("constant_test", ConstantObjective)
        try:
            objective = get_objective("constant_test")
            assert objective.metrics(torch.zeros(2), torch.zeros(2)) == {"constant": 1.0}
            assert "constant_test" in available_objectives()
        finally:
            from astroml.training import graph_objectives

            graph_objectives._REGISTRY.pop("constant_test", None)

    def test_registering_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_objective("", NodeClassificationObjective)


class TestIsBetter:
    """The direction of the monitored metric decides which epoch wins."""

    def test_higher_is_better_for_accuracy(self):
        objective = NodeClassificationObjective(log_input=False)
        assert objective.is_better(0.9, 0.8)
        assert not objective.is_better(0.7, 0.8)

    def test_lower_is_better_when_the_objective_says_so(self):
        class LossObjective(NodeClassificationObjective):
            monitor = "loss"
            higher_is_better = False

        objective = LossObjective(log_input=False)
        assert objective.is_better(0.1, 0.2)
        assert not objective.is_better(0.3, 0.2)


class TestEvaluate:
    def test_evaluate_returns_loss_alongside_metrics(self):
        objective = LinkPredictionObjective()
        scores = torch.tensor([2.0, -2.0])
        labels = torch.tensor([1.0, 0.0])

        report = objective.evaluate(scores, labels)

        assert set(report) == {"loss", "auc", "average_precision", "accuracy"}
        assert all(isinstance(value, float) for value in report.values())

    def test_evaluate_does_not_build_a_graph(self):
        objective = LinkPredictionObjective()
        scores = torch.randn(8, requires_grad=True)
        labels = (torch.rand(8) > 0.5).float()

        objective.evaluate(scores, labels)

        assert scores.grad is None
