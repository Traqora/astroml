from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

import numpy as np
import pytest
import torch

from astroml.models.temporal import (
    TemporalAttention,
    TemporalEncoding,
    TemporalGAT,
    TemporalGCN,
    TemporalGraphSAGE,
    TemporalGraphTransformer,
    TemporalModelFactory,
)


class TestTemporalEncoding:
    """Test temporal encoding functionality."""

    def test_temporal_encoding_init(self) -> Any:
        """Test temporal encoding initialization."""
        encoder = TemporalEncoding(temporal_dim=32)
        assert encoder.temporal_dim == 32
        assert encoder.max_time == 1000.0

    def test_temporal_encoding_forward(self) -> Any:
        """Test temporal encoding forward pass."""
        encoder = TemporalEncoding(temporal_dim=16)
        timestamps = torch.tensor([1.0, 2.0, 3.0, 4.0])

        encoding = encoder(timestamps)
        assert encoding.shape == (4, 16)
        assert torch.isfinite(encoding).all()

    def test_temporal_encoding_different_dims(self) -> Any:
        """Test temporal encoding with different dimensions."""
        for dim in [8, 16, 32, 64]:
            encoder = TemporalEncoding(temporal_dim=dim)
            timestamps = torch.rand(10)
            encoding = encoder(timestamps)
            assert encoding.shape == (10, dim)


class TestTemporalAttention:
    """Test temporal attention functionality."""

    def test_temporal_attention_init(self) -> Any:
        """Test temporal attention initialization."""
        attention = TemporalAttention(input_dim=64, temporal_dim=32, heads=8)
        assert attention.input_dim == 64
        assert attention.temporal_dim == 32
        assert attention.heads == 8

    def test_temporal_attention_forward(self) -> Any:
        """Test temporal attention forward pass."""
        attention = TemporalAttention(input_dim=64, temporal_dim=32, heads=4)

        x = torch.randn(10, 64)
        temporal_encoding = torch.randn(10, 32)

        output = attention(x, temporal_encoding)
        assert output.shape == (10, 64)
        assert torch.isfinite(output).all()

    def test_temporal_attention_residual(self) -> Any:
        """Test residual connection in temporal attention."""
        attention = TemporalAttention(input_dim=32, temporal_dim=16, heads=4)

        x = torch.randn(5, 32)
        temporal_encoding = torch.randn(5, 16)

        output = attention(x, temporal_encoding)
        # Output should be different from input due to attention
        assert not torch.equal(output, x)


class TestTemporalGCN:
    """Test TemporalGCN functionality."""

    def test_temporal_gcn_init(self) -> Any:
        """Test TemporalGCN initialization."""
        model = TemporalGCN(
            input_dim=64,
            hidden_dims=[32, 16],
            output_dim=2,
            temporal_dim=32
        )

        assert model.input_dim == 64
        assert model.hidden_dims == [32, 16]
        assert model.output_dim == 2
        assert model.temporal_dim == 32

    def test_temporal_gcn_forward_basic(self) -> Any:
        """Test TemporalGCN basic forward pass."""
        model = TemporalGCN(
            input_dim=16,
            hidden_dims=[32],
            output_dim=2,
            temporal_dim=8
        )

        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        output = model(x, edge_index)
        assert output.shape == (10, 2)
        assert torch.isfinite(output).all()

    def test_temporal_gcn_forward_with_time(self) -> Any:
        """Test TemporalGCN forward pass with temporal information."""
        model = TemporalGCN(
            input_dim=16,
            hidden_dims=[32],
            output_dim=2,
            temporal_dim=8
        )

        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        node_time = torch.rand(10)
        edge_time = torch.rand(3)

        output = model(x, edge_index, node_time=node_time, edge_time=edge_time)
        assert output.shape == (10, 2)
        assert torch.isfinite(output).all()

    def test_temporal_gcn_different_encodings(self) -> Any:
        """Test TemporalGCN with different time encodings."""
        x = torch.randn(5, 16)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        for encoding in ["sinusoidal", "learnable"]:
            model = TemporalGCN(
                input_dim=16,
                hidden_dims=[32],
                output_dim=2,
                temporal_dim=8,
                time_encoding=encoding
            )

            output = model(x, edge_index)
            assert output.shape == (5, 2)
            assert torch.isfinite(output).all()


class TestTemporalGraphSAGE:
    """Test TemporalGraphSAGE functionality."""

    def test_temporal_sage_init(self) -> Any:
        """Test TemporalGraphSAGE initialization."""
        model = TemporalGraphSAGE(
            input_dim=64,
            hidden_dims=[32, 16],
            output_dim=2,
            temporal_dim=32
        )

        assert model.input_dim == 64
        assert model.hidden_dims == [32, 16]
        assert model.output_dim == 2

    def test_temporal_sage_forward(self) -> Any:
        """Test TemporalGraphSAGE forward pass."""
        model = TemporalGraphSAGE(
            input_dim=16,
            hidden_dims=[32],
            output_dim=2,
            temporal_dim=8
        )

        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        node_time = torch.rand(10)

        output = model(x, edge_index, node_time=node_time)
        assert output.shape == (10, 2)
        assert torch.isfinite(output).all()


class TestTemporalGAT:
    """Test TemporalGAT functionality."""

    def test_temporal_gat_init(self) -> Any:
        """Test TemporalGAT initialization."""
        model = TemporalGAT(
            input_dim=64,
            hidden_dims=[32, 16],
            output_dim=2,
            temporal_dim=32,
            heads=8
        )

        assert model.input_dim == 64
        assert model.heads == 8

    def test_temporal_gat_forward(self) -> Any:
        """Test TemporalGAT forward pass."""
        model = TemporalGAT(
            input_dim=16,
            hidden_dims=[32],
            output_dim=2,
            temporal_dim=8,
            heads=4
        )

        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        node_time = torch.rand(10)

        output = model(x, edge_index, node_time=node_time)
        assert output.shape == (10, 2)
        assert torch.isfinite(output).all()


class TestTemporalGraphTransformer:
    """Test TemporalGraphTransformer functionality."""

    def test_temporal_transformer_init(self) -> Any:
        """Test TemporalGraphTransformer initialization."""
        model = TemporalGraphTransformer(
            input_dim=64,
            hidden_dim=128,
            output_dim=2,
            temporal_dim=32,
            num_heads=8,
            num_layers=2
        )

        assert model.input_dim == 64
        assert model.hidden_dim == 128
        assert model.num_heads == 8
        assert model.num_layers == 2

    def test_temporal_transformer_forward(self) -> Any:
        """Test TemporalGraphTransformer forward pass."""
        model = TemporalGraphTransformer(
            input_dim=16,
            hidden_dim=32,
            output_dim=2,
            temporal_dim=8,
            num_heads=4,
            num_layers=2
        )

        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        node_time = torch.rand(10)

        output = model(x, edge_index, node_time=node_time)
        assert output.shape == (10, 2)
        assert torch.isfinite(output).all()


class TestTemporalModelFactory:
    """Test TemporalModelFactory functionality."""

    def test_create_temporal_gcn(self) -> Any:
        """Test creating TemporalGCN from factory."""
        config = {
            'input_dim': 64,
            'hidden_dims': [32, 16],
            'output_dim': 2,
            'temporal_dim': 32,
            'dropout': 0.5
        }

        model = TemporalModelFactory.create_temporal_gcn(config)
        assert isinstance(model, TemporalGCN)
        assert model.input_dim == 64
        assert model.output_dim == 2

    def test_create_temporal_sage(self) -> Any:
        """Test creating TemporalGraphSAGE from factory."""
        config = {
            'input_dim': 64,
            'hidden_dims': [32, 16],
            'output_dim': 2,
            'temporal_dim': 32,
            'dropout': 0.5
        }

        model = TemporalModelFactory.create_temporal_sage(config)
        assert isinstance(model, TemporalGraphSAGE)
        assert model.input_dim == 64

    def test_create_temporal_gat(self) -> Any:
        """Test creating TemporalGAT from factory."""
        config = {
            'input_dim': 64,
            'hidden_dims': [32, 16],
            'output_dim': 2,
            'temporal_dim': 32,
            'heads': 8
        }

        model = TemporalModelFactory.create_temporal_gat(config)
        assert isinstance(model, TemporalGAT)
        assert model.heads == 8

    def test_create_temporal_transformer(self) -> Any:
        """Test creating TemporalGraphTransformer from factory."""
        config = {
            'input_dim': 64,
            'hidden_dim': 128,
            'output_dim': 2,
            'temporal_dim': 32,
            'num_heads': 8,
            'num_layers': 3
        }

        model = TemporalModelFactory.create_temporal_transformer(config)
        assert isinstance(model, TemporalGraphTransformer)
        assert model.hidden_dim == 128


class TestTemporalModelIntegration:
    """Test integration of temporal models."""

    @pytest.fixture
    def sample_graph_data(self) -> Any:
        """Create sample graph data for testing."""
        return {
            'x': torch.randn(20, 16),
            'edge_index': torch.tensor([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
            ], dtype=torch.long),
            'node_time': torch.rand(20),
            'edge_time': torch.rand(10)
        }

    def test_all_models_forward(self, sample_graph_data) -> Any:
        """Test all temporal models can do forward pass."""
        models = [
            TemporalGCN(16, [32], 2, temporal_dim=8),
            TemporalGraphSAGE(16, [32], 2, temporal_dim=8),
            TemporalGAT(16, [32], 2, temporal_dim=8, heads=4),
            TemporalGraphTransformer(16, 32, 2, temporal_dim=8, num_heads=4, num_layers=2)
        ]

        for model in models:
            output = model(
                sample_graph_data['x'],
                sample_graph_data['edge_index'],
                node_time=sample_graph_data['node_time'],
                edge_time=sample_graph_data['edge_time']
            )

            assert output.shape == (20, 2)
            assert torch.isfinite(output).all()

            # Check log_softmax property
            assert torch.allclose(output.exp().sum(dim=1), torch.ones(20))

    def test_models_without_time(self, sample_graph_data) -> Any:
        """Test models work without temporal information."""
        models = [
            TemporalGCN(16, [32], 2, temporal_dim=8),
            TemporalGraphSAGE(16, [32], 2, temporal_dim=8),
            TemporalGAT(16, [32], 2, temporal_dim=8, heads=4)
        ]

        for model in models:
            output = model(
                sample_graph_data['x'],
                sample_graph_data['edge_index']
            )

            assert output.shape == (20, 2)
            assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__])
