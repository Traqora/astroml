"""Graph model input: isolated nodes, empty edge lists and bad tensors (issue #1200)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from astroml.models.gcn import GCN  # noqa: E402
from astroml.models.graph_input import isolated_node_mask, prepare_graph_input  # noqa: E402


def _edges(*pairs: tuple[int, int]) -> "torch.Tensor":
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class TestIsolatedNodeMask:
    def test_marks_nodes_with_no_edges(self) -> None:
        mask = isolated_node_mask(_edges((0, 1), (1, 0)), num_nodes=4)
        assert mask.tolist() == [False, False, True, True]

    def test_a_node_whose_only_edge_is_a_self_loop_is_isolated(self) -> None:
        mask = isolated_node_mask(_edges((0, 1), (1, 0), (2, 2)), num_nodes=3)
        assert mask.tolist() == [False, False, True]

    def test_every_node_is_isolated_without_edges(self) -> None:
        mask = isolated_node_mask(torch.empty((2, 0), dtype=torch.long), num_nodes=3)
        assert mask.tolist() == [True, True, True]

    def test_a_directed_edge_connects_both_endpoints(self) -> None:
        mask = isolated_node_mask(_edges((0, 2)), num_nodes=3)
        assert mask.tolist() == [False, True, False]


class TestPrepareGraphInput:
    def test_counts_isolated_nodes(self) -> None:
        prepared = prepare_graph_input(torch.randn(5, 4), _edges((0, 1), (1, 0), (2, 2)))
        assert prepared.num_nodes == 5
        assert prepared.num_edges == 3
        assert prepared.num_isolated == 3
        assert prepared.isolated_fraction == pytest.approx(0.6)

    def test_an_empty_float_edge_list_is_accepted_as_long(self) -> None:
        # ``torch.tensor([[], []])`` is float32, the natural way to write "no edges".
        empty = torch.tensor([[], []])
        assert empty.dtype == torch.float32

        prepared = prepare_graph_input(torch.randn(3, 4), empty)

        assert prepared.edge_index.dtype == torch.long
        assert prepared.edge_index.shape == (2, 0)
        assert prepared.num_edges == 0
        assert prepared.num_isolated == 3

    def test_int32_edges_are_converted_to_long(self) -> None:
        edges = _edges((0, 1), (1, 0)).to(torch.int32)
        assert prepare_graph_input(torch.randn(2, 4), edges).edge_index.dtype == torch.long

    def test_a_graph_with_no_nodes_is_valid(self) -> None:
        prepared = prepare_graph_input(torch.empty((0, 4)), torch.tensor([[], []]))
        assert prepared.num_nodes == 0
        assert prepared.isolated_fraction == 0.0

    def test_rejects_a_non_empty_float_edge_list(self) -> None:
        with pytest.raises(ValueError, match="integer indices"):
            prepare_graph_input(torch.randn(3, 4), torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    def test_rejects_an_index_past_the_last_node(self) -> None:
        with pytest.raises(ValueError, match=r"outside \[0, 3\).*max=3"):
            prepare_graph_input(torch.randn(3, 4), _edges((0, 3)))

    def test_rejects_a_negative_index(self) -> None:
        with pytest.raises(ValueError, match="min=-1"):
            prepare_graph_input(torch.randn(3, 4), _edges((-1, 0)))

    def test_rejects_non_finite_features_and_names_the_node(self) -> None:
        x = torch.randn(4, 3)
        x[2, 1] = float("nan")
        with pytest.raises(ValueError, match="node 2"):
            prepare_graph_input(x, _edges((0, 1)))

    def test_rejects_infinite_features(self) -> None:
        x = torch.randn(3, 3)
        x[0, 0] = float("inf")
        with pytest.raises(ValueError, match="NaN or infinite"):
            prepare_graph_input(x, _edges((0, 1)))

    def test_rejects_integer_features(self) -> None:
        with pytest.raises(ValueError, match="floating point"):
            prepare_graph_input(torch.ones(3, 2, dtype=torch.long), _edges((0, 1)))

    @pytest.mark.parametrize("shape", [(3,), (3, 2, 1)])
    def test_rejects_features_that_are_not_a_matrix(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"shape \[N, F\]"):
            prepare_graph_input(torch.randn(*shape), _edges((0, 1)))

    def test_rejects_an_edge_index_that_is_not_two_rows(self) -> None:
        with pytest.raises(ValueError, match=r"shape \[2, E\]"):
            prepare_graph_input(torch.randn(3, 4), torch.zeros((3, 2), dtype=torch.long))


class TestGCNWithIsolatedNodes:
    """The model's own behaviour on the inputs the helper accepts."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = GCN(4, 8, 3).eval()

    def test_isolated_nodes_give_finite_output(self) -> None:
        x = torch.randn(5, 4)
        out = self.model(x, _edges((0, 1), (1, 0), (2, 3), (3, 2)))
        assert out.shape == (5, 3)
        assert bool(torch.isfinite(out).all())

    def test_an_isolated_node_gets_the_output_it_would_have_alone(self) -> None:
        x = torch.randn(5, 4)
        in_graph = self.model(x, _edges((0, 1), (1, 0), (2, 3), (3, 2)))
        alone = self.model(x[4:5], torch.empty((2, 0), dtype=torch.long))
        assert torch.allclose(in_graph[4:5], alone, atol=1e-6)

    def test_a_graph_with_no_edges_runs_even_when_the_empty_list_is_float(self) -> None:
        # Regression: this used to fail in the scatter kernel with
        # "Expected dtype int32/int64 for index".
        out = self.model(torch.randn(4, 4), torch.tensor([[], []]))
        assert out.shape == (4, 3)
        assert bool(torch.isfinite(out).all())

    def test_isolated_nodes_do_not_break_training(self) -> None:
        self.model.train()
        x = torch.randn(6, 4)
        edges = _edges((0, 1), (1, 0))
        loss = torch.nn.functional.nll_loss(self.model(x, edges), torch.tensor([0, 1, 2, 0, 1, 2]))
        loss.backward()
        grads = [p.grad for p in self.model.parameters()]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
