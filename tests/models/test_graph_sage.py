"""GraphSAGE model variant and its training wiring (issue #736).

Covers the three things that make it a usable alternative to the GCN: the
aggregation variants actually aggregate differently, the API is
interchangeable with :class:`astroml.models.gcn.GCN`, and a malformed
adjacency is refused rather than trained on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from astroml.models.graph_sage import (  # noqa: E402
    AGGREGATIONS,
    GraphSAGE,
    SAGEAggregator,
    aggregate_neighbours,
    validate_edge_index,
)
from astroml.training.train_graph_sage import (  # noqa: E402
    GraphSAGEConfig,
    evaluate,
    train_graph_sage,
)


def _toy_graph(num_nodes: int = 24, num_features: int = 6, seed: int = 0):
    """A graph whose labels are linearly separable from the features.

    Separable on purpose: a training test should fail because the wiring
    broke, not because the task was hard.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, num_features, generator=generator)
    y = (x[:, 0] > 0).long()

    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[: num_nodes // 2] = True
    val_mask = ~train_mask
    return x, edge_index, y, train_mask, val_mask


class TestAggregation:
    def test_mean_averages_the_incoming_messages(self):
        messages = torch.tensor([[2.0], [4.0], [10.0]])
        destinations = torch.tensor([0, 0, 1])

        out = aggregate_neighbours(messages, destinations, 2, "mean")

        assert out.tolist() == [[3.0], [10.0]]

    def test_sum_adds_them(self):
        messages = torch.tensor([[2.0], [4.0]])
        destinations = torch.tensor([0, 0])

        assert aggregate_neighbours(messages, destinations, 1, "sum").tolist() == [[6.0]]

    def test_max_takes_the_largest(self):
        messages = torch.tensor([[2.0], [7.0], [4.0]])
        destinations = torch.tensor([0, 0, 0])

        assert aggregate_neighbours(messages, destinations, 1, "max").tolist() == [[7.0]]

    def test_gcn_counts_the_node_as_one_of_its_own_neighbours(self):
        messages = torch.tensor([[3.0], [3.0]])
        destinations = torch.tensor([0, 0])

        # Two neighbours, divisor of three.
        assert aggregate_neighbours(messages, destinations, 1, "gcn").tolist() == [[2.0]]

    def test_a_node_with_no_incoming_edges_aggregates_to_zero(self):
        messages = torch.tensor([[5.0]])
        destinations = torch.tensor([0])

        for strategy in AGGREGATIONS:
            out = aggregate_neighbours(messages, destinations, 3, strategy)
            assert out[1].tolist() == [0.0], f"{strategy} left node 1 non-zero"
            assert out[2].tolist() == [0.0], f"{strategy} left node 2 non-zero"

    def test_max_does_not_clamp_negative_messages_to_zero(self):
        messages = torch.tensor([[-5.0], [-2.0]])
        destinations = torch.tensor([0, 0])

        # A regression guard: initialising the scatter buffer at zero and
        # forgetting include_self would report 0.0 here.
        assert aggregate_neighbours(messages, destinations, 1, "max").tolist() == [[-2.0]]

    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown aggregation"):
            aggregate_neighbours(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long), 1, "median")

    def test_no_edges_produces_a_zero_matrix(self):
        out = aggregate_neighbours(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), 4, "mean")

        assert out.shape == (4, 3)
        assert float(out.abs().sum()) == 0.0


class TestEdgeIndexValidation:
    def test_a_valid_index_is_returned_as_long(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

        assert validate_edge_index(edge_index, 2).dtype == torch.long

    def test_the_wrong_shape_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[2, E\]"):
            validate_edge_index(torch.zeros(3, 4, dtype=torch.long), 4)

    def test_an_out_of_range_node_is_rejected(self):
        # Silently wrapping this into a valid row is how a GNN ends up
        # trained on an adjacency that does not exist.
        with pytest.raises(ValueError, match="but the graph has 2 nodes"):
            validate_edge_index(torch.tensor([[0], [5]]), 2)

    def test_a_negative_index_is_rejected(self):
        with pytest.raises(ValueError, match="node -1"):
            validate_edge_index(torch.tensor([[0], [-1]]), 4)

    def test_a_non_tensor_is_rejected(self):
        with pytest.raises(TypeError, match="must be a Tensor"):
            validate_edge_index([[0], [1]], 2)

    def test_an_empty_index_over_a_real_graph_is_fine(self):
        assert validate_edge_index(torch.empty(2, 0, dtype=torch.long), 5).numel() == 0


class TestModelApi:
    def test_it_matches_the_gcn_constructor_and_output_contract(self):
        x, edge_index, _, _, _ = _toy_graph()
        model = GraphSAGE(input_dim=6, hidden_dim=16, output_dim=2, dropout=0.5)

        out = model(x, edge_index)

        assert out.shape == (x.size(0), 2)
        # log_softmax, exactly like GCN.forward.
        assert torch.allclose(out.exp().sum(dim=1), torch.ones(x.size(0)), atol=1e-5)

    @pytest.mark.parametrize("aggregator", AGGREGATIONS)
    def test_every_aggregator_produces_a_usable_model(self, aggregator):
        x, edge_index, _, _, _ = _toy_graph()

        out = GraphSAGE(6, 8, 3, aggregator=aggregator)(x, edge_index)

        assert out.shape == (x.size(0), 3)
        assert torch.isfinite(out).all()

    def test_hidden_dims_configures_each_layer_independently(self):
        model = GraphSAGE(6, 16, 2, hidden_dims=[32, 16, 8])

        assert model.num_layers == 4
        assert [conv.out_dim for conv in model.convs] == [32, 16, 8, 2]

    def test_num_layers_repeats_hidden_dim(self):
        model = GraphSAGE(6, 16, 2, num_layers=3)

        assert [conv.out_dim for conv in model.convs] == [16, 16, 2]

    def test_a_single_layer_model_maps_straight_to_the_classes(self):
        model = GraphSAGE(6, 16, 2, num_layers=1)

        assert model.num_layers == 1
        assert model.convs[0].out_dim == 2

    def test_embed_returns_logits_without_the_softmax(self):
        x, edge_index, _, _, _ = _toy_graph()
        model = GraphSAGE(6, 8, 4).eval()

        logits = model.embed(x, edge_index)

        assert logits.shape == (x.size(0), 4)
        assert not torch.allclose(logits.exp().sum(dim=1), torch.ones(x.size(0)), atol=1e-3)

    def test_a_graph_with_no_edges_still_classifies(self):
        x, _, _, _, _ = _toy_graph()

        out = GraphSAGE(6, 8, 2)(x, torch.empty(2, 0, dtype=torch.long))

        assert out.shape == (x.size(0), 2)
        assert torch.isfinite(out).all()

    def test_gradients_reach_every_parameter(self):
        x, edge_index, y, train_mask, _ = _toy_graph()
        model = GraphSAGE(6, 8, 2)

        torch.nn.functional.nll_loss(model(x, edge_index)[train_mask], y[train_mask]).backward()

        unused = [name for name, p in model.named_parameters() if p.grad is None]
        assert not unused, f"parameters with no gradient: {unused}"

    def test_the_gcn_variant_has_no_dead_self_transform(self):
        # gcn folds the node into its own mean, so a separate self weight
        # would sit in the optimiser with no gradient forever.
        assert SAGEAggregator(4, 4, aggregation="gcn").lin_self is None
        assert SAGEAggregator(4, 4, aggregation="mean").lin_self is not None

    def test_it_is_interchangeable_with_the_gcn_signature(self):
        import inspect

        from astroml.models.gcn import GCN

        gcn_args = list(inspect.signature(GCN.__init__).parameters)[1:5]
        sage_args = list(inspect.signature(GraphSAGE.__init__).parameters)[1:5]

        assert gcn_args == sage_args


class TestModelValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"input_dim": 0}, "input_dim and output_dim"),
            ({"output_dim": 0}, "input_dim and output_dim"),
            ({"dropout": 1.0}, "dropout"),
            ({"dropout": -0.1}, "dropout"),
            ({"aggregator": "median"}, "unknown aggregator"),
            ({"num_layers": 0}, "num_layers"),
            ({"hidden_dim": 0}, "hidden_dim"),
            ({"hidden_dims": [4, 0]}, "hidden_dims"),
        ],
    )
    def test_bad_configuration_is_rejected_at_construction(self, kwargs, match):
        base = {"input_dim": 6, "hidden_dim": 8, "output_dim": 2}
        with pytest.raises(ValueError, match=match):
            GraphSAGE(**{**base, **kwargs})

    def test_one_dimensional_features_are_rejected(self):
        model = GraphSAGE(6, 8, 2)

        with pytest.raises(ValueError, match=r"\[N, F\]"):
            model(torch.randn(6), torch.empty(2, 0, dtype=torch.long))


class TestTraining:
    def test_training_reduces_the_loss(self):
        x, edge_index, y, train_mask, val_mask = _toy_graph()

        _, history = train_graph_sage(
            x, edge_index, y, train_mask, val_mask, GraphSAGEConfig(epochs=60, hidden_dim=16)
        )

        assert history.train_loss[-1] < history.train_loss[0]

    def test_it_learns_a_separable_task(self):
        x, edge_index, y, train_mask, val_mask = _toy_graph()

        model, _ = train_graph_sage(
            x, edge_index, y, train_mask, val_mask, GraphSAGEConfig(epochs=120, hidden_dim=16)
        )

        assert evaluate(model, x, edge_index, y, val_mask)["accuracy"] > 0.7

    def test_the_same_seed_reproduces_the_run(self):
        x, edge_index, y, train_mask, val_mask = _toy_graph()
        config = GraphSAGEConfig(epochs=25, seed=11)

        _, first = train_graph_sage(x, edge_index, y, train_mask, val_mask, config)
        _, second = train_graph_sage(x, edge_index, y, train_mask, val_mask, config)

        assert first.train_loss == second.train_loss

    def test_a_different_seed_changes_the_run(self):
        x, edge_index, y, train_mask, val_mask = _toy_graph()

        _, a = train_graph_sage(
            x, edge_index, y, train_mask, val_mask, GraphSAGEConfig(epochs=25, seed=1)
        )
        _, b = train_graph_sage(
            x, edge_index, y, train_mask, val_mask, GraphSAGEConfig(epochs=25, seed=2)
        )

        assert a.train_loss != b.train_loss

    def test_the_best_validation_weights_are_returned(self):
        x, edge_index, y, train_mask, val_mask = _toy_graph()

        model, history = train_graph_sage(
            x, edge_index, y, train_mask, val_mask, GraphSAGEConfig(epochs=80)
        )

        # Training past the optimum must not hand back the worse final model.
        assert evaluate(model, x, edge_index, y, val_mask)["accuracy"] == pytest.approx(
            history.best_val_accuracy
        )

    def test_early_stopping_ends_the_run(self):
        x, edge_index, y, train_mask, val_mask = _toy_graph()

        _, history = train_graph_sage(
            x,
            edge_index,
            y,
            train_mask,
            val_mask,
            GraphSAGEConfig(epochs=500, early_stopping_patience=3),
        )

        assert history.stopped_early
        assert history.epochs_run < 500

    def test_training_without_a_validation_mask_runs_the_full_schedule(self):
        x, edge_index, y, train_mask, _ = _toy_graph()

        _, history = train_graph_sage(
            x, edge_index, y, train_mask, config=GraphSAGEConfig(epochs=10)
        )

        assert history.epochs_run == 10
        assert history.val_accuracy == []

    def test_the_model_is_left_in_a_usable_eval_state(self):
        x, edge_index, y, train_mask, val_mask = _toy_graph()

        model, _ = train_graph_sage(
            x, edge_index, y, train_mask, val_mask, GraphSAGEConfig(epochs=5)
        )

        first = evaluate(model, x, edge_index, y, val_mask)
        second = evaluate(model, x, edge_index, y, val_mask)
        # Dropout must be off during evaluation, so two passes agree.
        assert first == second

    def test_history_serialises(self):
        x, edge_index, y, train_mask, val_mask = _toy_graph()

        _, history = train_graph_sage(
            x, edge_index, y, train_mask, val_mask, GraphSAGEConfig(epochs=5)
        )

        assert set(history.to_dict()) >= {"train_loss", "val_accuracy", "epochs_run"}


class TestTrainingValidation:
    def test_an_empty_train_mask_is_rejected(self):
        x, edge_index, y, _, _ = _toy_graph()

        with pytest.raises(ValueError, match="selects no nodes"):
            train_graph_sage(x, edge_index, y, torch.zeros(x.size(0), dtype=torch.bool))

    def test_a_non_boolean_mask_is_rejected(self):
        x, edge_index, y, _, _ = _toy_graph()

        with pytest.raises(ValueError, match="bool tensor"):
            train_graph_sage(x, edge_index, y, torch.ones(x.size(0), dtype=torch.long))

    def test_a_mask_of_the_wrong_length_is_rejected(self):
        x, edge_index, y, _, _ = _toy_graph()

        with pytest.raises(ValueError, match="must cover all"):
            train_graph_sage(x, edge_index, y, torch.ones(3, dtype=torch.bool))

    def test_mismatched_labels_are_rejected(self):
        x, edge_index, _, train_mask, _ = _toy_graph()

        with pytest.raises(ValueError, match="y must have shape"):
            train_graph_sage(x, edge_index, torch.zeros(3, dtype=torch.long), train_mask)

    def test_an_out_of_range_edge_is_rejected_before_training(self):
        x, _, y, train_mask, _ = _toy_graph()

        with pytest.raises(ValueError, match="but the graph has"):
            train_graph_sage(x, torch.tensor([[0], [999]]), y, train_mask)

    def test_a_bad_config_is_rejected(self):
        x, edge_index, y, train_mask, _ = _toy_graph()

        with pytest.raises(ValueError, match="epochs"):
            train_graph_sage(x, edge_index, y, train_mask, config=GraphSAGEConfig(epochs=0))


class TestRegistration:
    def test_it_is_exported_from_the_models_package(self):
        from astroml import models

        assert models.GraphSAGE is GraphSAGE
        assert "GraphSAGE" in models.__all__

    def test_it_imports_without_torch_geometric(self):
        import sys

        import astroml.models.graph_sage as module

        # The GCN needs PyG; this one deliberately does not, so it stays
        # importable in a deployment that does not ship it.
        assert "torch_geometric" not in getattr(module, "__dict__", {})
        assert module.__name__ in sys.modules
