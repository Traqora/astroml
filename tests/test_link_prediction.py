"""Tests for self-supervised link prediction task and model."""

from __future__ import annotations

import json
import random

import pytest
import torch

from astroml.features.graph.snapshot import Edge
from astroml.tasks.link_prediction_task import (
    LedgerSplit,
    LinkPredictionTask,
    sample_negative_edges,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edges(n: int = 20, max_ledger: int = 10) -> list[Edge]:
    """Return n edges with ledger timestamps spread across max_ledger values."""
    rng = random.Random(42)
    accounts = [f"account_{i}" for i in range(6)]
    edges = []
    for i in range(n):
        src = rng.choice(accounts)
        dst = rng.choice([a for a in accounts if a != src])
        ts = rng.randint(1, max_ledger)
        edges.append(Edge(src=src, dst=dst, timestamp=ts))
    return edges


# ---------------------------------------------------------------------------
# LedgerSplit
# ---------------------------------------------------------------------------


class TestLedgerSplit:
    def test_num_nodes(self):
        edges = _make_edges(10)
        accounts = {e.src for e in edges} | {e.dst for e in edges}
        node_index = {a: i for i, a in enumerate(sorted(accounts))}
        split = LedgerSplit(context_edges=edges, future_edges=[], node_index=node_index)
        assert split.num_nodes == len(accounts)

    def test_to_edge_index_shape(self):
        edges = [Edge("a", "b", 1), Edge("b", "c", 2)]
        node_index = {"a": 0, "b": 1, "c": 2}
        split = LedgerSplit(context_edges=edges, future_edges=[], node_index=node_index)
        ei = split.to_edge_index(edges)
        assert ei.shape == (2, 2)
        assert ei.dtype == torch.long

    def test_to_edge_index_unknown_accounts_skipped(self):
        edges = [Edge("a", "b", 1), Edge("x", "y", 2)]  # x, y not in index
        node_index = {"a": 0, "b": 1}
        split = LedgerSplit(context_edges=edges, future_edges=[], node_index=node_index)
        ei = split.to_edge_index(edges)
        assert ei.shape == (2, 1)  # only the a→b edge

    def test_to_edge_index_empty(self):
        node_index = {"a": 0, "b": 1}
        split = LedgerSplit(context_edges=[], future_edges=[], node_index=node_index)
        ei = split.to_edge_index([])
        assert ei.shape == (2, 0)


# ---------------------------------------------------------------------------
# sample_negative_edges
# ---------------------------------------------------------------------------


class TestSampleNegativeEdges:
    def test_no_positives_in_negatives(self):
        pos_set = {(0, 1), (1, 2)}
        negs = sample_negative_edges(
            num_nodes=5, positive_set=pos_set, num_samples=10, rng=random.Random(0)
        )
        for u, v in negs:
            assert (u, v) not in pos_set

    def test_no_self_loops(self):
        negs = sample_negative_edges(
            num_nodes=10, positive_set=set(), num_samples=20, rng=random.Random(0)
        )
        for u, v in negs:
            assert u != v

    def test_returns_at_most_num_samples(self):
        negs = sample_negative_edges(
            num_nodes=4, positive_set=set(), num_samples=100, rng=random.Random(0)
        )
        # 4 nodes → at most 4*3=12 directed non-self-loop pairs
        assert len(negs) <= 12

    def test_uniqueness(self):
        negs = sample_negative_edges(
            num_nodes=20, positive_set=set(), num_samples=30, rng=random.Random(1)
        )
        assert len(negs) == len(set(negs))

    def test_empty_and_saturated_graphs_return_no_negatives(self):
        assert sample_negative_edges(0, set(), 10, rng=random.Random(0)) == []
        saturated_pairs = {(u, v) for u in range(3) for v in range(3) if u != v}
        assert sample_negative_edges(3, saturated_pairs, 10, rng=random.Random(0)) == []

    def test_dense_graph_returns_only_feasible_negative_pairs(self):
        positives = {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2)}
        negatives = sample_negative_edges(
            num_nodes=3,
            positive_set=positives,
            num_samples=10,
            max_attempts=100,
            rng=random.Random(3),
        )
        assert negatives == [(2, 1)]

    def test_invalid_sampling_bounds_raise(self):
        with pytest.raises(ValueError, match="num_nodes"):
            sample_negative_edges(-1, set(), 1)
        with pytest.raises(ValueError, match="num_samples"):
            sample_negative_edges(2, set(), -1)


class TestSyntheticLedgerDataset:
    def test_generation_is_reproducible_and_metadata_persists(self, tmp_path):
        from astroml.training.synthetic_dataset import generate_synthetic_ledger_dataset

        dataset = generate_synthetic_ledger_dataset(
            num_accounts=5, num_edges=15, num_ledgers=8, feature_dim=4, seed=10
        )
        repeated = generate_synthetic_ledger_dataset(
            num_accounts=5, num_edges=15, num_ledgers=8, feature_dim=4, seed=10
        )

        assert dataset.edges == repeated.edges
        assert torch.equal(dataset.node_features, repeated.node_features)
        assert dataset.features_for({"account_1": 0, "account_0": 1}).shape == (2, 4)

        metadata_path = dataset.save_metadata(tmp_path / "metadata" / "dataset.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["node_count"] == 5
        assert metadata["edge_count"] == 15
        assert metadata["first_timestamp"] == min(edge.timestamp for edge in dataset.edges)
        assert metadata["last_timestamp"] == max(edge.timestamp for edge in dataset.edges)
        assert metadata["seed"] == 10
        assert dataset.lineage.start.timestamp() == 1
        assert dataset.lineage.end.timestamp() == 8

    def test_empty_dataset_has_empty_bounds_and_rejects_impossible_edges(self):
        from astroml.training.synthetic_dataset import generate_synthetic_ledger_dataset

        dataset = generate_synthetic_ledger_dataset(
            num_accounts=0, num_edges=0, num_ledgers=1, feature_dim=4
        )
        assert dataset.node_features.shape == (0, 4)
        assert dataset.lineage.first_timestamp is None
        assert dataset.lineage.last_timestamp is None

        with pytest.raises(ValueError, match="at least two accounts"):
            generate_synthetic_ledger_dataset(num_accounts=1, num_edges=1)


class _FixedScoreModel:
    def eval(self):
        return self

    def encode(self, node_features, edge_index):
        return node_features.new_zeros((node_features.shape[0], 1))

    def decode(self, embeddings, edge_index):
        return embeddings.new_zeros((edge_index.shape[1],))


class TestLinkPredictionEvaluationBoundaries:
    def test_empty_evaluation_returns_defined_metrics(self):
        task = LinkPredictionTask([], n_future=1)
        split = LedgerSplit(context_edges=[], future_edges=[], node_index={})

        assert task.evaluate(_FixedScoreModel(), split, torch.empty((0, 1))) == {
            "auc": 0.5,
            "avg_precision": 0.0,
        }

    def test_single_class_evaluation_returns_defined_metrics(self):
        task = LinkPredictionTask([], n_future=1)
        positive_edges = [Edge("a", "b", 1), Edge("b", "a", 1)]
        split = LedgerSplit(
            context_edges=[],
            future_edges=positive_edges,
            node_index={"a": 0, "b": 1},
        )

        assert task.evaluate(_FixedScoreModel(), split, torch.zeros((2, 1))) == {
            "auc": 0.5,
            "avg_precision": 1.0,
        }

    def test_two_class_evaluation_returns_ranked_metrics(self):
        task = LinkPredictionTask([], n_future=1, seed=0)
        split = LedgerSplit(
            context_edges=[],
            future_edges=[Edge("a", "b", 1)],
            node_index={"a": 0, "b": 1, "c": 2},
        )

        assert task.evaluate(_FixedScoreModel(), split, torch.zeros((3, 1))) == {
            "auc": 0.5,
            "avg_precision": 0.5,
        }


# ---------------------------------------------------------------------------
# LinkPredictionTask.build_splits
# ---------------------------------------------------------------------------


class TestBuildSplits:
    def test_splits_are_produced(self):
        edges = _make_edges(30, max_ledger=15)
        task = LinkPredictionTask(edges, n_future=3, seed=0)
        splits = task.build_splits()
        assert len(splits) > 0

    def test_context_strictly_before_future(self):
        edges = _make_edges(30, max_ledger=15)
        task = LinkPredictionTask(edges, n_future=3, seed=0)
        for split in task.build_splits():
            ctx_max = max(e.timestamp for e in split.context_edges)
            fut_min = min(e.timestamp for e in split.future_edges)
            assert ctx_max < fut_min, f"Context max={ctx_max} is not < future min={fut_min}"

    def test_future_window_bounded_by_n_future(self):
        edges = _make_edges(40, max_ledger=20)
        n_future = 4
        task = LinkPredictionTask(edges, n_future=n_future, seed=0)
        for split in task.build_splits():
            fut_min = min(e.timestamp for e in split.future_edges)
            fut_max = max(e.timestamp for e in split.future_edges)
            assert fut_max < fut_min + n_future

    def test_context_ledgers_restricts_window(self):
        edges = _make_edges(50, max_ledger=20)
        task = LinkPredictionTask(edges, n_future=3, context_ledgers=2, seed=0)
        for split in task.build_splits():
            ctx_ledgers = {e.timestamp for e in split.context_edges}
            assert len(ctx_ledgers) <= 2

    def test_empty_edges_returns_no_splits(self):
        task = LinkPredictionTask([], n_future=5, seed=0)
        assert task.build_splits() == []

    def test_node_index_covers_all_accounts(self):
        edges = _make_edges(20, max_ledger=10)
        task = LinkPredictionTask(edges, n_future=3, seed=0)
        for split in task.build_splits():
            all_accounts = (
                {e.src for e in split.context_edges}
                | {e.dst for e in split.context_edges}
                | {e.src for e in split.future_edges}
                | {e.dst for e in split.future_edges}
            )
            assert all_accounts <= set(split.node_index.keys())

    def test_invalid_n_future_raises(self):
        with pytest.raises(ValueError, match="n_future"):
            LinkPredictionTask(_make_edges(), n_future=0)

    def test_invalid_neg_ratio_raises(self):
        with pytest.raises(ValueError, match="neg_sampling_ratio"):
            LinkPredictionTask(_make_edges(), neg_sampling_ratio=0.0)


# ---------------------------------------------------------------------------
# LinkPredictionTask.sample_negatives
# ---------------------------------------------------------------------------


class TestSampleNegatives:
    def _make_split(self) -> LedgerSplit:
        edges = [Edge("a", "b", 1), Edge("b", "c", 1)]
        future = [Edge("a", "c", 2)]
        node_index = {"a": 0, "b": 1, "c": 2}
        return LedgerSplit(context_edges=edges, future_edges=future, node_index=node_index)

    def test_returns_tensor_shape(self):
        task = LinkPredictionTask(_make_edges(), n_future=3, seed=0)
        split = self._make_split()
        neg = task.sample_negatives(split)
        assert neg.dim() == 2
        assert neg.shape[0] == 2

    def test_negatives_not_in_future_edges(self):
        task = LinkPredictionTask(_make_edges(), n_future=3, seed=42)
        split = self._make_split()
        neg = task.sample_negatives(split)
        future_pairs = {(0, 2)}  # a→c in node_index
        for i in range(neg.size(1)):
            assert (neg[0, i].item(), neg[1, i].item()) not in future_pairs


# ---------------------------------------------------------------------------
# LinkPredictor model (unit tests — no GCN forward, just decoder logic)
# ---------------------------------------------------------------------------


class TestLinkPredictorDecoder:
    """Test the decoder and loss without running a full GCN forward."""

    def _dummy_embeddings(self, n=8, dim=16) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(n, dim)

    def test_dot_decoder_shape(self):
        try:
            from astroml.models.link_prediction import LinkPredictor
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model = LinkPredictor(input_dim=8, hidden_dims=[16], embedding_dim=16, decoder="dot")
        z = self._dummy_embeddings()
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        scores = model.decode(z, edge_index)
        assert scores.shape == (3,)

    def test_mlp_decoder_shape(self):
        try:
            from astroml.models.link_prediction import LinkPredictor
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model = LinkPredictor(input_dim=8, hidden_dims=[16], embedding_dim=16, decoder="mlp")
        z = self._dummy_embeddings()
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        scores = model.decode(z, edge_index)
        assert scores.shape == (2,)

    def test_loss_is_scalar(self):
        try:
            from astroml.models.link_prediction import LinkPredictor
        except ImportError:
            pytest.skip("torch_geometric not installed")

        model = LinkPredictor(input_dim=8, hidden_dims=[16], embedding_dim=16)
        z = self._dummy_embeddings()
        pos = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        neg = torch.tensor([[0, 2], [3, 4]], dtype=torch.long)
        loss = model.loss(z, pos, neg)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_decreases_with_training(self):
        try:
            from astroml.models.link_prediction import LinkPredictor
        except ImportError:
            pytest.skip("torch_geometric not installed")

        torch.manual_seed(1)
        model = LinkPredictor(input_dim=8, hidden_dims=[16], embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Fixed embeddings simulating encoder output
        z = self._dummy_embeddings()
        pos = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        neg = torch.tensor([[4, 5, 6], [5, 6, 7]], dtype=torch.long)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = model.loss(z, pos, neg)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease over training steps"
