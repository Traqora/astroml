import torch

from astroml.utils.temporal import TemporalGraphBuilder


def _transactions():
    return [
        {
            "source_account": "alice",
            "target_account": "bob",
            "timestamp": 10.0,
            "amount": 2.5,
            "operation_type": "payment",
        },
        {
            "source_account": "bob",
            "target_account": "carol",
            "timestamp": 20.0,
            "amount": 7.0,
            "operation_type": "path_payment",
        },
        {
            "source_account": "alice",
            "target_account": "carol",
            "timestamp": 30.0,
            "amount": 1.5,
        },
    ]


def test_build_temporal_graph_vectorized_edges_and_features():
    graph = TemporalGraphBuilder().build_temporal_graph(_transactions())
    mapping = graph["node_mapping"]

    expected_edges = torch.tensor(
        [
            [mapping["alice"], mapping["bob"], mapping["alice"]],
            [mapping["bob"], mapping["carol"], mapping["carol"]],
        ],
        dtype=torch.long,
    )

    assert torch.equal(graph["edge_index"], expected_edges)
    assert torch.allclose(graph["edge_weights"], torch.tensor([2.5, 7.0, 1.5]))
    assert torch.allclose(graph["edge_features"][:, 0], torch.tensor([2.5, 7.0, 1.5]))
    assert graph["edge_features"].shape == (3, 2)
    assert graph["num_nodes"] == 3


def test_build_temporal_graph_vectorized_node_feature_totals():
    graph = TemporalGraphBuilder().build_temporal_graph(_transactions())
    mapping = graph["node_mapping"]
    base_features = TemporalGraphBuilder()._create_basic_node_features(
        list(mapping.keys()), _transactions()
    )

    alice = mapping["alice"]
    bob = mapping["bob"]
    carol = mapping["carol"]

    assert torch.allclose(base_features[alice], torch.tensor([2.0, 4.0, 0.0, -4.0]))
    assert torch.allclose(base_features[bob], torch.tensor([2.0, 7.0, 2.5, -4.5]))
    assert torch.allclose(base_features[carol], torch.tensor([2.0, 0.0, 8.5, 8.5]))
    assert graph["node_features"].shape == (3, 13)


def test_build_temporal_graph_empty_input():
    graph = TemporalGraphBuilder().build_temporal_graph([])

    assert graph["edge_index"].shape == (2, 0)
    assert graph["edge_features"].shape == (0, 2)
    assert graph["node_features"].shape == (0, 0)
    assert graph["node_mapping"] == {}
