"""Unit tests for Graph Neural Network (GNN) models and training pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astroml.graph_utils import (
    build_transaction_pyg_graph,
    compute_graph_metrics,
    extract_k_hop_subgraph,
    generate_node_embeddings,
    graph_to_pyg_data,
)
from astroml.training.gnn.data_loader import (
    GraphDataPreprocessor,
    TransactionGraphDataLoader,
    TransactionGraphDataset,
    create_node_masks,
)
from astroml.training.gnn.gat import (
    GATNodeClassifier,
    evaluate_gat,
    train_gat_classifier,
)
from astroml.training.gnn.graph_sage import (
    GraphSAGENodeClassifier,
    evaluate_graphsage,
    train_graphsage_classifier,
)


@pytest.fixture
def sample_graph():
    """Synthetic graph fixture for testing GNN architectures."""
    torch.manual_seed(42)
    num_nodes = 20
    in_channels = 8
    x = torch.randn(num_nodes, in_channels)

    # Connected ring + shortcuts
    src = list(range(num_nodes)) + [0, 2, 4, 6, 8]
    dst = [(i + 1) % num_nodes for i in range(num_nodes)] + [5, 7, 9, 11, 13]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    labels = torch.randint(0, 2, (num_nodes,))
    return x, edge_index, labels


class TestGraphSAGE:
    """Tests for GraphSAGENodeClassifier architecture and training."""

    def test_model_initialization_and_forward(self, sample_graph):
        x, edge_index, _ = sample_graph
        model = GraphSAGENodeClassifier(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
            dropout=0.2,
        )

        out = model(x, edge_index)
        assert out.shape == (20, 2)
        assert not torch.isnan(out).any()

    def test_deep_sage_architecture(self, sample_graph):
        x, edge_index, _ = sample_graph
        model = GraphSAGENodeClassifier(
            in_channels=8,
            hidden_channels=[32, 16],
            out_channels=3,
            num_layers=3,
            use_batch_norm=True,
            normalize_embeddings=True,
        )
        out = model(x, edge_index)
        assert out.shape == (20, 3)

    def test_predict_and_proba(self, sample_graph):
        x, edge_index, _ = sample_graph
        model = GraphSAGENodeClassifier(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
        )
        probs = model.predict_proba(x, edge_index)
        preds = model.predict(x, edge_index)

        assert probs.shape == (20, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(20), atol=1e-5)
        assert preds.shape == (20,)
        assert (preds >= 0).all() and (preds < 2).all()

    def test_get_embeddings(self, sample_graph):
        x, edge_index, _ = sample_graph
        model = GraphSAGENodeClassifier(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
        )
        emb = model.get_embeddings(x, edge_index, layer=0)
        assert emb.shape == (20, 16)

    def test_training_and_evaluation_loop(self, sample_graph):
        x, edge_index, labels = sample_graph
        train_mask = torch.zeros(20, dtype=torch.bool)
        train_mask[:14] = True
        val_mask = torch.zeros(20, dtype=torch.bool)
        val_mask[14:] = True

        model = GraphSAGENodeClassifier(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
        )

        res = train_graphsage_classifier(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            epochs=15,
            lr=0.05,
        )

        assert res["epochs_trained"] == 15
        assert res["final_train_loss"] >= 0.0
        assert 0.0 <= res["final_train_acc"] <= 1.0

        eval_res = evaluate_graphsage(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            mask=val_mask,
        )
        assert "accuracy" in eval_res
        assert "loss" in eval_res
        assert "f1_score" in eval_res


class TestGAT:
    """Tests for GATNodeClassifier architecture and training."""

    def test_gat_forward_and_attention(self, sample_graph):
        x, edge_index, _ = sample_graph
        model = GATNodeClassifier(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            heads=2,
            num_layers=2,
        )

        out = model(x, edge_index)
        assert out.shape == (20, 2)

        attn = model.get_attention_weights(x, edge_index)
        if attn is not None:
            _, alpha = attn
            assert alpha.dim() >= 1

    def test_gat_residual_and_predict(self, sample_graph):
        x, edge_index, _ = sample_graph
        model = GATNodeClassifier(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            heads=2,
            num_layers=2,
            residual=True,
        )

        probs = model.predict_proba(x, edge_index)
        preds = model.predict(x, edge_index)
        assert probs.shape == (20, 2)
        assert preds.shape == (20,)

        embs = model.get_embeddings(x, edge_index)
        assert embs.size(0) == 20

    def test_gat_training_and_evaluation(self, sample_graph):
        x, edge_index, labels = sample_graph
        train_mask = torch.zeros(20, dtype=torch.bool)
        train_mask[:15] = True
        val_mask = torch.zeros(20, dtype=torch.bool)
        val_mask[15:] = True

        model = GATNodeClassifier(
            in_channels=8,
            hidden_channels=16,
            out_channels=2,
            heads=2,
            num_layers=2,
        )

        res = train_gat_classifier(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            epochs=10,
            lr=0.01,
        )

        assert res["epochs_trained"] == 10
        assert "final_train_loss" in res

        eval_res = evaluate_gat(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            mask=val_mask,
        )
        assert 0.0 <= eval_res["accuracy"] <= 1.0


class TestDataLoaderAndPreprocessor:
    """Tests for GraphDataPreprocessor, masks, and transaction loaders."""

    def test_preprocessor(self):
        feats = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
        prep = GraphDataPreprocessor(normalize_features=True)
        transformed = prep.fit_transform(feats)

        assert transformed.shape == (3, 2)
        assert torch.allclose(transformed.mean(dim=0), torch.zeros(2), atol=1e-5)

    def test_create_node_masks(self):
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        t_mask, v_mask, te_mask = create_node_masks(
            num_nodes=10,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            labels=labels,
            stratify=True,
        )

        assert (t_mask | v_mask | te_mask).all()
        assert (t_mask & v_mask).sum() == 0
        assert (t_mask & te_mask).sum() == 0

    def test_transaction_graph_dataset_and_loader(self, sample_graph):
        x, edge_index, labels = sample_graph
        dataset = TransactionGraphDataset(
            x=x,
            edge_index=edge_index,
            y=labels,
        )
        assert len(dataset) == 20
        item = dataset[0]
        assert item["node_idx"] == 0
        assert "feature" in item

        loader = TransactionGraphDataLoader(
            x=x,
            edge_index=edge_index,
            y=labels,
            batch_size=8,
            num_neighbors=[5, 5],
        )
        batches = list(loader)
        assert len(batches) > 0
        batch = batches[0]
        assert "target_nodes" in batch
        assert "sampled_nodes" in batch
        assert "adjs" in batch


class TestGraphUtils:
    """Tests for graph construction and analysis utilities."""

    def test_build_transaction_pyg_graph(self):
        transactions = [
            {
                "source_account": "acc_A",
                "destination_account": "acc_B",
                "amount": 100.0,
                "fee": 0.1,
            },
            {
                "source_account": "acc_B",
                "destination_account": "acc_C",
                "amount": 50.0,
                "fee": 0.05,
            },
            {
                "source_account": "acc_C",
                "destination_account": "acc_A",
                "amount": 25.0,
                "fee": 0.01,
            },
        ]
        data, node_to_id = build_transaction_pyg_graph(transactions)

        assert len(node_to_id) == 3
        assert data.num_nodes == 3
        assert data.edge_index.shape == (2, 3)
        assert data.x.shape == (3, 8)

    def test_extract_k_hop_subgraph(self, sample_graph):
        x, edge_index, _ = sample_graph
        subset, sub_edges, mapping, edge_mask = extract_k_hop_subgraph(
            node_idx=0,
            num_hops=1,
            edge_index=edge_index,
            num_nodes=20,
        )
        assert len(subset) >= 1
        assert sub_edges.shape[0] == 2

    def test_generate_node_embeddings(self, sample_graph):
        x, edge_index, labels = sample_graph
        data = graph_to_pyg_data(node_features=x.numpy(), edge_index=edge_index.numpy())
        model = GraphSAGENodeClassifier(in_channels=8, hidden_channels=16, out_channels=4)

        embeddings = generate_node_embeddings(model, data)
        assert embeddings.shape == (20, 4)

    def test_compute_graph_metrics(self, sample_graph):
        x, edge_index, _ = sample_graph
        data = graph_to_pyg_data(node_features=x.numpy(), edge_index=edge_index.numpy())
        metrics = compute_graph_metrics(data)

        assert metrics["num_nodes"] == 20
        assert metrics["num_edges"] == edge_index.size(1)
        assert "density" in metrics
        assert "avg_degree" in metrics
