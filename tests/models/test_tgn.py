"""Temporal Graph Network (issue #737).

What distinguishes a TGN from the static temporal models already here is that
memory carries between snapshots. So the tests concentrate on the state: that
it updates, that it is reset when it should be, that it is *not* changed by
an evaluation pass, and that replaying the same stream twice gives the same
answer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from astroml.models.tgn import (  # noqa: E402
    MESSAGE_AGGREGATIONS,
    MemoryState,
    TemporalGraphNetwork,
    TimeEncoder,
    aggregate_messages,
    validate_temporal_batch,
)
from astroml.training.train_tgn import (  # noqa: E402
    TGNConfig,
    evaluate_tgn,
    train_tgn,
    validate_snapshots,
)

NUM_NODES = 18
NUM_FEATURES = 5


def _snapshot(start_time: float, seed: int, num_edges: int = 36) -> dict:
    """One snapshot whose labels are separable from the node features."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(NUM_NODES, NUM_FEATURES, generator=generator)
    y = (x[:, 0] > 0).long()
    edge_index = torch.stack(
        [
            torch.randint(0, NUM_NODES, (num_edges,), generator=generator),
            torch.randint(0, NUM_NODES, (num_edges,), generator=generator),
        ]
    )
    edge_time = torch.linspace(start_time, start_time + 90, num_edges)
    return {"x": x, "y": y, "edge_index": edge_index, "edge_time": edge_time}


def _stream(count: int = 3, first_seed: int = 0) -> list[dict]:
    return [_snapshot(index * 100.0, first_seed + index) for index in range(count)]


def _model(**kwargs) -> TemporalGraphNetwork:
    defaults = {
        "input_dim": NUM_FEATURES,
        "hidden_dim": 16,
        "output_dim": 2,
        "memory_dim": 8,
        "time_dim": 8,
        "message_dim": 8,
        "num_nodes": NUM_NODES,
    }
    torch.manual_seed(0)
    return TemporalGraphNetwork(**{**defaults, **kwargs})


class TestTimeEncoder:
    def test_it_produces_the_requested_width(self):
        encoded = TimeEncoder(12)(torch.tensor([0.0, 10.0, 100.0]))

        assert encoded.shape == (3, 12)

    def test_output_is_bounded(self):
        encoded = TimeEncoder(8)(torch.tensor([0.0, 1e6]))

        # A cosine basis: unbounded output would mean the projection broke.
        assert float(encoded.min()) >= -1.0
        assert float(encoded.max()) <= 1.0

    def test_different_elapsed_times_encode_differently(self):
        encoder = TimeEncoder(16)
        encoded = encoder(torch.tensor([0.0, 5.0, 5000.0]))

        assert not torch.allclose(encoded[0], encoded[1])
        assert not torch.allclose(encoded[1], encoded[2])

    def test_it_accepts_both_shapes(self):
        encoder = TimeEncoder(8)

        flat = encoder(torch.tensor([1.0, 2.0]))
        column = encoder(torch.tensor([[1.0], [2.0]]))

        assert torch.allclose(flat, column)

    def test_a_zero_width_encoder_is_rejected(self):
        with pytest.raises(ValueError, match="dimension"):
            TimeEncoder(0)


class TestMessageAggregation:
    def test_mean_averages_messages_to_the_same_node(self):
        messages = torch.tensor([[2.0], [4.0], [10.0]])
        destinations = torch.tensor([0, 0, 1])

        out = aggregate_messages(messages, destinations, 2, "mean")

        assert out.tolist() == [[3.0], [10.0]]

    def test_sum_adds_them(self):
        out = aggregate_messages(torch.tensor([[2.0], [4.0]]), torch.tensor([0, 0]), 1, "sum")

        assert out.tolist() == [[6.0]]

    def test_last_keeps_the_final_message_in_order(self):
        messages = torch.tensor([[2.0], [9.0]])
        destinations = torch.tensor([0, 0])

        out = aggregate_messages(messages, destinations, 1, "last")

        assert out.tolist() == [[9.0]]

    def test_a_node_with_no_messages_stays_zero(self):
        for aggregation in MESSAGE_AGGREGATIONS:
            out = aggregate_messages(torch.tensor([[5.0]]), torch.tensor([0]), 3, aggregation)
            assert out[1].tolist() == [0.0]
            assert out[2].tolist() == [0.0]

    def test_an_unknown_aggregation_is_rejected(self):
        with pytest.raises(ValueError, match="unknown aggregation"):
            aggregate_messages(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long), 1, "median")

    def test_no_messages_gives_a_zero_matrix(self):
        out = aggregate_messages(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), 4, "mean")

        assert out.shape == (4, 3)
        assert float(out.abs().sum()) == 0.0


class TestBatchValidation:
    def test_a_valid_batch_passes(self):
        edge_index = torch.tensor([[0, 1], [1, 2]])

        assert validate_temporal_batch(edge_index, torch.tensor([1.0, 2.0]), None, 0).dtype is (
            torch.int64
        )

    def test_a_length_mismatch_is_rejected(self):
        # The classic silent failure: timestamps that do not correspond to
        # the edges they are supposed to date.
        with pytest.raises(ValueError, match="edge_time must have shape"):
            validate_temporal_batch(torch.tensor([[0, 1], [1, 2]]), torch.tensor([1.0]), None, 0)

    def test_the_wrong_edge_index_shape_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[2, E\]"):
            validate_temporal_batch(torch.zeros(3, 4, dtype=torch.long), torch.zeros(4), None, 0)

    def test_negative_node_indices_are_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            validate_temporal_batch(torch.tensor([[0], [-1]]), torch.tensor([1.0]), None, 0)

    def test_missing_edge_features_are_rejected_when_expected(self):
        with pytest.raises(ValueError, match="no edge_attr"):
            validate_temporal_batch(torch.tensor([[0], [1]]), torch.tensor([1.0]), None, 4)

    def test_edge_features_of_the_wrong_width_are_rejected(self):
        with pytest.raises(ValueError, match="expects 4"):
            validate_temporal_batch(
                torch.tensor([[0], [1]]), torch.tensor([1.0]), torch.zeros(1, 3), 4
            )


class TestStandardInterface:
    def test_forward_matches_the_temporal_model_contract(self):
        model = _model()
        snapshot = _snapshot(0.0, seed=1)

        out = model(snapshot["x"], snapshot["edge_index"], edge_time=snapshot["edge_time"])

        assert out.shape == (NUM_NODES, 2)
        assert torch.allclose(out.exp().sum(dim=1), torch.ones(NUM_NODES), atol=1e-5)

    def test_it_accepts_the_same_keyword_arguments_as_the_other_temporal_models(self):
        import inspect

        from astroml.models.temporal import TemporalGCN

        shared = {"x", "edge_index", "edge_time", "node_time", "edge_attr"}
        tgn_args = set(inspect.signature(TemporalGraphNetwork.forward).parameters) - {"self"}
        gcn_args = set(inspect.signature(TemporalGCN.forward).parameters) - {"self"}

        assert shared <= tgn_args
        assert shared <= gcn_args

    def test_node_time_is_accepted_and_ignored(self):
        # eval() so dropout does not make the two passes differ for reasons
        # that have nothing to do with node_time.
        model = _model().eval()
        snapshot = _snapshot(0.0, seed=1)

        model.reset_memory()
        with_time = model(
            snapshot["x"],
            snapshot["edge_index"],
            edge_time=snapshot["edge_time"],
            node_time=torch.zeros(NUM_NODES),
        )
        model.reset_memory()
        without = model(snapshot["x"], snapshot["edge_index"], edge_time=snapshot["edge_time"])

        assert torch.allclose(with_time, without)

    def test_missing_edge_times_default_to_zero(self):
        model = _model()
        snapshot = _snapshot(0.0, seed=1)

        out = model(snapshot["x"], snapshot["edge_index"])

        assert out.shape == (NUM_NODES, 2)

    def test_edge_features_are_consumed_when_configured(self):
        model = _model(edge_dim=3)
        snapshot = _snapshot(0.0, seed=1)
        edge_attr = torch.randn(snapshot["edge_index"].size(1), 3)

        out = model(
            snapshot["x"],
            snapshot["edge_index"],
            edge_time=snapshot["edge_time"],
            edge_attr=edge_attr,
        )

        assert out.shape == (NUM_NODES, 2)

    def test_a_snapshot_with_no_interactions_still_scores(self):
        model = _model()
        snapshot = _snapshot(0.0, seed=1)

        out = model(snapshot["x"], torch.empty(2, 0, dtype=torch.long))

        assert out.shape == (NUM_NODES, 2)
        assert torch.isfinite(out).all()

    def test_wrong_feature_width_is_rejected(self):
        model = _model()

        with pytest.raises(ValueError, match="expects 5"):
            model(torch.randn(NUM_NODES, 3), torch.empty(2, 0, dtype=torch.long))

    def test_gradients_reach_every_parameter(self):
        model = _model()
        snapshot = _snapshot(0.0, seed=1)

        out = model(snapshot["x"], snapshot["edge_index"], edge_time=snapshot["edge_time"])
        torch.nn.functional.nll_loss(out, snapshot["y"]).backward()

        unused = [name for name, p in model.named_parameters() if p.grad is None]
        assert not unused, f"parameters with no gradient: {unused}"


class TestModelValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"input_dim": 0}, "input_dim and output_dim"),
            ({"output_dim": 0}, "input_dim and output_dim"),
            ({"hidden_dim": 0}, "must all be >= 1"),
            ({"memory_dim": 0}, "must all be >= 1"),
            ({"edge_dim": -1}, "edge_dim"),
            ({"dropout": 1.0}, "dropout"),
            ({"message_aggregation": "median"}, "unknown message_aggregation"),
            ({"num_nodes": 0}, "num_nodes"),
        ],
    )
    def test_bad_configuration_is_rejected_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _model(**kwargs)


class TestMemory:
    def test_memory_starts_at_zero(self):
        model = _model()

        assert float(model.get_memory().memory.abs().sum()) == 0.0

    def test_an_interaction_changes_memory(self):
        model = _model()
        snapshot = _snapshot(0.0, seed=1)

        model(snapshot["x"], snapshot["edge_index"], edge_time=snapshot["edge_time"])

        assert float(model.get_memory().memory.abs().sum()) > 0.0

    def test_reset_clears_it(self):
        model = _model()
        snapshot = _snapshot(0.0, seed=1)
        model(snapshot["x"], snapshot["edge_index"], edge_time=snapshot["edge_time"])

        model.reset_memory()

        assert float(model.get_memory().memory.abs().sum()) == 0.0

    def test_only_nodes_that_interacted_are_updated(self):
        model = _model()
        x = torch.randn(NUM_NODES, NUM_FEATURES)
        # Only nodes 0 and 1 take part.
        model(x, torch.tensor([[0], [1]]), edge_time=torch.tensor([5.0]))

        memory = model.get_memory().memory
        assert float(memory[2:].abs().sum()) == 0.0, "an idle node's memory drifted"
        assert float(memory[:2].abs().sum()) > 0.0

    def test_last_update_records_the_interaction_time(self):
        model = _model()

        model(
            torch.randn(NUM_NODES, NUM_FEATURES),
            torch.tensor([[0], [1]]),
            edge_time=torch.tensor([77.0]),
        )

        last_update = model.get_memory().last_update
        assert float(last_update[0]) == 77.0
        assert float(last_update[1]) == 77.0
        assert float(last_update[2]) == 0.0

    def test_memory_carries_between_snapshots(self):
        model = _model()
        stream = _stream(2)

        first = model(stream[0]["x"], stream[0]["edge_index"], edge_time=stream[0]["edge_time"])
        model.reset_memory()
        fresh = model(stream[1]["x"], stream[1]["edge_index"], edge_time=stream[1]["edge_time"])

        model.reset_memory()
        model(stream[0]["x"], stream[0]["edge_index"], edge_time=stream[0]["edge_time"])
        carried = model(stream[1]["x"], stream[1]["edge_index"], edge_time=stream[1]["edge_time"])

        # The whole point of a TGN: the second snapshot is read differently
        # because the first one happened.
        assert not torch.allclose(carried, fresh)
        assert first.shape == carried.shape

    def test_using_memory_before_it_exists_is_an_error(self):
        model = TemporalGraphNetwork(input_dim=4, hidden_dim=8, output_dim=2)

        with pytest.raises(RuntimeError, match="not been initialised"):
            model.get_memory()

    def test_reset_without_a_size_is_an_error(self):
        model = TemporalGraphNetwork(input_dim=4, hidden_dim=8, output_dim=2)

        with pytest.raises(ValueError, match="num_nodes is unknown"):
            model.reset_memory()

    def test_a_batch_larger_than_memory_is_rejected(self):
        model = _model(num_nodes=4)

        with pytest.raises(ValueError, match="memory holds 4 nodes"):
            model(torch.randn(10, NUM_FEATURES), torch.tensor([[0], [9]]))

    def test_detach_keeps_the_values(self):
        model = _model()
        snapshot = _snapshot(0.0, seed=1)
        model(snapshot["x"], snapshot["edge_index"], edge_time=snapshot["edge_time"])
        before = model.get_memory().memory.detach().clone()

        model.detach_memory()

        assert torch.allclose(model.get_memory().memory, before)
        assert not model.get_memory().memory.requires_grad

    def test_a_cloned_state_is_independent(self):
        model = _model()
        snapshot = _snapshot(0.0, seed=1)
        model(snapshot["x"], snapshot["edge_index"], edge_time=snapshot["edge_time"])

        saved = model.get_memory().clone()
        model.reset_memory()

        assert float(saved.memory.abs().sum()) > 0.0
        assert isinstance(saved, MemoryState)


class TestDeterminism:
    def test_interaction_order_within_a_snapshot_does_not_matter(self):
        snapshot = _snapshot(0.0, seed=1)
        shuffle = torch.randperm(
            snapshot["edge_index"].size(1), generator=torch.Generator().manual_seed(5)
        )

        model = _model().eval()
        model.reset_memory()
        ordered = model(snapshot["x"], snapshot["edge_index"], edge_time=snapshot["edge_time"])

        model.reset_memory()
        shuffled = model(
            snapshot["x"],
            snapshot["edge_index"][:, shuffle],
            edge_time=snapshot["edge_time"][shuffle],
        )

        # Memory updates are order-dependent, so the batch is sorted into one
        # canonical sequence before anything is written.
        assert torch.allclose(ordered, shuffled, atol=1e-6)

    def test_replaying_a_stream_gives_the_same_answer(self):
        model = _model().eval()
        stream = _stream(3)

        first = model.forward_stream(stream)
        second = model.forward_stream(stream)

        for a, b in zip(first, second):
            assert torch.allclose(a, b)

    def test_two_models_with_the_same_seed_agree(self):
        stream = _stream(2)

        a = _model().eval().forward_stream(stream)
        b = _model().eval().forward_stream(stream)

        assert torch.allclose(a[-1], b[-1])


class TestStreaming:
    def test_it_returns_one_prediction_per_snapshot(self):
        model = _model()
        stream = _stream(4)

        outputs = model.forward_stream(stream)

        assert len(outputs) == 4
        assert all(out.shape == (NUM_NODES, 2) for out in outputs)

    def test_the_stream_resets_memory_by_default(self):
        model = _model()
        stream = _stream(2)
        model.forward_stream(stream)
        after_first_run = model.get_memory().memory.clone()

        model.forward_stream(stream)

        # A second run over the same stream must not inherit the first run's
        # state, or the two are not comparable.
        assert torch.allclose(model.get_memory().memory, after_first_run)

    def test_detaching_between_snapshots_bounds_the_graph(self):
        model = _model()

        model.forward_stream(_stream(3), detach_between=True)

        assert not model.get_memory().memory.requires_grad


class TestSnapshotValidation:
    def test_an_empty_stream_is_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_snapshots([])

    def test_a_snapshot_missing_labels_is_rejected(self):
        snapshot = _snapshot(0.0, seed=1)
        del snapshot["y"]

        with pytest.raises(ValueError, match="missing 'y'"):
            validate_snapshots([snapshot])

    def test_mismatched_labels_are_rejected(self):
        snapshot = _snapshot(0.0, seed=1)
        snapshot["y"] = torch.zeros(3, dtype=torch.long)

        with pytest.raises(ValueError, match="y must have shape"):
            validate_snapshots([snapshot])

    def test_an_out_of_order_stream_is_rejected(self):
        stream = _stream(3)

        # Replaying a temporal model out of order gives a confidently wrong
        # answer rather than an error, so refuse it up front.
        with pytest.raises(ValueError, match="must be chronological"):
            validate_snapshots(list(reversed(stream)))

    def test_a_non_boolean_mask_is_rejected(self):
        snapshot = _snapshot(0.0, seed=1)
        snapshot["mask"] = torch.ones(NUM_NODES, dtype=torch.long)

        with pytest.raises(ValueError, match="mask must be a bool"):
            validate_snapshots([snapshot])

    def test_it_reports_the_node_count_the_stream_spans(self):
        assert validate_snapshots(_stream(3)) == NUM_NODES


class TestTraining:
    def test_a_smoke_run_completes_and_reduces_the_loss(self):
        stream = _stream(3)

        _, history = train_tgn(stream, config=TGNConfig(epochs=20, hidden_dim=16, memory_dim=8))

        assert history.epochs_run == 20
        assert history.train_loss[-1] < history.train_loss[0]

    def test_it_learns_a_separable_task(self):
        train = _stream(3)
        validation = _stream(2, first_seed=50)

        model, _ = train_tgn(
            train,
            val_snapshots=validation,
            config=TGNConfig(epochs=40, hidden_dim=32, memory_dim=16),
        )

        assert evaluate_tgn(model, validation)["accuracy"] > 0.6

    def test_the_same_seed_reproduces_the_run(self):
        stream = _stream(2)
        config = TGNConfig(epochs=8, seed=11)

        _, first = train_tgn(stream, config=config)
        _, second = train_tgn(stream, config=config)

        assert first.train_loss == second.train_loss

    def test_a_different_seed_changes_the_run(self):
        stream = _stream(2)

        _, a = train_tgn(stream, config=TGNConfig(epochs=8, seed=1))
        _, b = train_tgn(stream, config=TGNConfig(epochs=8, seed=2))

        assert a.train_loss != b.train_loss

    def test_early_stopping_ends_the_run(self):
        train = _stream(2)
        validation = _stream(2, first_seed=90)

        _, history = train_tgn(
            train,
            val_snapshots=validation,
            config=TGNConfig(epochs=300, early_stopping_patience=2),
        )

        assert history.stopped_early
        assert history.epochs_run < 300

    def test_every_epoch_starts_from_a_clean_memory(self):
        stream = _stream(2)
        config = TGNConfig(epochs=4, seed=7)

        _, one_epoch = train_tgn(stream, config=TGNConfig(epochs=1, seed=7))
        _, four_epochs = train_tgn(stream, config=config)

        # The first epoch of the longer run must match the whole short run;
        # if memory leaked across epochs it would not.
        assert four_epochs.train_loss[0] == pytest.approx(one_epoch.train_loss[0])

    def test_masks_restrict_the_supervised_nodes(self):
        stream = _stream(2)
        for snapshot in stream:
            mask = torch.zeros(NUM_NODES, dtype=torch.bool)
            mask[:6] = True
            snapshot["mask"] = mask

        _, history = train_tgn(stream, config=TGNConfig(epochs=5))

        assert history.epochs_run == 5

    def test_a_bad_config_is_rejected(self):
        with pytest.raises(ValueError, match="epochs"):
            train_tgn(_stream(2), config=TGNConfig(epochs=0))

    def test_history_serialises(self):
        _, history = train_tgn(_stream(2), config=TGNConfig(epochs=3))

        assert set(history.to_dict()) >= {"train_loss", "epochs_run", "duration_seconds"}


class TestEvaluation:
    def test_evaluation_reports_loss_and_accuracy(self):
        model = _model()
        stream = _stream(2)

        result = evaluate_tgn(model, stream)

        assert set(result) == {"loss", "accuracy", "num_nodes"}
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluating_does_not_change_memory(self):
        model = _model()
        stream = _stream(2)
        model.forward_stream(stream)
        before = model.get_memory().memory.clone()

        evaluate_tgn(model, stream)

        # An eval pass that advanced memory would move the model somewhere
        # the training config never described.
        assert torch.allclose(model.get_memory().memory, before)

    def test_evaluating_twice_gives_the_same_answer(self):
        model = _model()
        stream = _stream(2)

        assert evaluate_tgn(model, stream) == evaluate_tgn(model, stream)

    def test_evaluation_leaves_training_mode_untouched(self):
        model = _model()
        model.train()

        evaluate_tgn(model, _stream(1))

        assert model.training


class TestRegistration:
    def test_it_is_exported_from_the_models_package(self):
        from astroml import models

        assert models.TemporalGraphNetwork is TemporalGraphNetwork
        assert "TemporalGraphNetwork" in models.__all__

    def test_the_temporal_factory_can_build_it(self):
        from astroml.models.temporal import TemporalModelFactory

        class _Config(dict):
            input_dim = NUM_FEATURES
            hidden_dim = 16
            output_dim = 2

        model = TemporalModelFactory.create_tgn(_Config())

        assert isinstance(model, TemporalGraphNetwork)

    def test_it_imports_without_torch_geometric(self):
        import astroml.models.tgn as module

        # temporal.py needs PyG; the TGN deliberately does not, so it stays
        # importable in a deployment that does not ship it.
        assert "torch_geometric" not in module.__dict__
