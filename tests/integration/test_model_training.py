"""Integration tests for the model training pipeline.

These tests verify the complete workflow from features to trained models,
including training, evaluation, and model persistence.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from astroml.models.gcn import GCN
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.training.train_sage import train_epoch, build_reconstruction_target
from astroml.features.gnn.sampler import MultiHopSampler


class TestGCNTrainingIntegration:
    """Integration tests for GCN model training."""

    def test_gcn_training_workflow(
        self,
        sample_training_data: tuple,
    ) -> None:
        """Test complete GCN training workflow."""
        X, y = sample_training_data
        
        # Create simple graph structure (random edges)
        num_nodes = X.shape[0]
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        
        # Convert to PyG format
        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
        )
        
        # Create model
        model = GCN(
            input_dim=X.shape[1],
            hidden_dim=16,
            output_dim=2,
            dropout=0.5,
        )
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()
        
        # Train for a few epochs
        model.train()
        initial_loss = None
        for epoch in range(5):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            if epoch == 0:
                initial_loss = loss.item()
        
        # Verify loss decreased
        final_loss = loss.item()
        assert final_loss < initial_loss or final_loss == initial_loss

    def test_gcn_prediction_workflow(
        self,
        sample_training_data: tuple,
    ) -> None:
        """Test GCN prediction workflow after training."""
        X, y = sample_training_data
        num_nodes = X.shape[0]
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        
        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
        )
        
        model = GCN(
            input_dim=X.shape[1],
            hidden_dim=16,
            output_dim=2,
            dropout=0.0,  # No dropout for prediction
        )
        
        # Train briefly
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        
        # Predict
        model.eval()
        with torch.no_grad():
            predictions = model(data.x, data.edge_index)
            predicted_classes = predictions.argmax(dim=1)
        
        # Verify predictions
        assert predicted_classes.shape == (num_nodes,)
        assert torch.all(predicted_classes >= 0)
        assert torch.all(predicted_classes < 2)


class TestGraphSAGETrainingIntegration:
    """Integration tests for GraphSAGE model training."""

    def test_sage_encoder_training(
        self,
        sample_node_features: Dict[str, np.ndarray],
        sample_edge_list: List[tuple],
    ) -> None:
        """Test GraphSAGE encoder training with reconstruction loss."""
        # Prepare data
        node_ids = list(sample_node_features.keys())
        features = np.stack([sample_node_features[nid] for nid in node_ids])
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Create edge index
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_list = []
        for src, dst, _, _ in sample_edge_list:
            if src in node_to_idx and dst in node_to_idx:
                edge_list.append([node_to_idx[src], node_to_idx[dst]])
        
        if len(edge_list) == 0:
            # Create dummy edges if none exist
            edge_list = [[0, 1], [1, 2], [2, 0]]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Create encoder
        encoder = InductiveSAGEEncoder(
            input_dim=features.shape[1],
            hidden_dim=16,
            output_dim=8,
            num_layers=2,
            dropout=0.0,
            aggregator='mean',
        )
        
        # Create sampler
        sampler = MultiHopSampler(edge_index, num_hops=2, fanout=[5, 5])
        
        # Train nodes
        train_nodes = torch.arange(min(10, len(node_ids)))
        
        # Training setup
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
        
        # Train for one epoch
        loss = train_epoch(
            encoder=encoder,
            sampler=sampler,
            features=features_tensor,
            edge_index=edge_index,
            train_nodes=train_nodes,
            optimizer=optimizer,
            batch_size=4,
            device='cpu',
        )
        
        # Verify loss is finite
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_reconstruction_target_computation(
        self,
        sample_node_features: Dict[str, np.ndarray],
        sample_edge_list: List[tuple],
    ) -> None:
        """Test reconstruction target computation for training."""
        node_ids = list(sample_node_features.keys())
        features = np.stack([sample_node_features[nid] for nid in node_ids])
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Create edge index
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_list = []
        for src, dst, _, _ in sample_edge_list:
            if src in node_to_idx and dst in node_to_idx:
                edge_list.append([node_to_idx[src], node_to_idx[dst]])
        
        if len(edge_list) == 0:
            edge_list = [[0, 1], [1, 2], [2, 0]]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Compute reconstruction targets
        target_nodes = torch.arange(min(5, len(node_ids)))
        targets = build_reconstruction_target(
            edge_index=edge_index,
            features=features_tensor,
            target_nodes=target_nodes,
        )
        
        # Verify shape and values
        assert targets.shape == (len(target_nodes), features.shape[1])
        assert torch.all(torch.isfinite(targets))


class TestModelPersistenceIntegration:
    """Integration tests for model persistence and loading."""

    def test_save_and_load_gcn_model(
        self,
        sample_training_data: tuple,
        temp_output_dir: Path,
    ) -> None:
        """Test saving and loading GCN model."""
        X, y = sample_training_data
        num_nodes = X.shape[0]
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        
        # Create and train model
        model = GCN(
            input_dim=X.shape[1],
            hidden_dim=16,
            output_dim=2,
            dropout=0.5,
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            data = Data(
                x=torch.tensor(X, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.long),
            )
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        
        # Save model
        model_path = temp_output_dir / "gcn_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'input_dim': X.shape[1],
            'hidden_dim': 16,
            'output_dim': 2,
        }, model_path)
        
        # Verify file exists
        assert model_path.exists()
        
        # Load model
        checkpoint = torch.load(model_path)
        loaded_model = GCN(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
        )
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify loaded model works
        loaded_model.eval()
        with torch.no_grad():
            data = Data(
                x=torch.tensor(X, dtype=torch.float32),
                edge_index=edge_index,
            )
            predictions = loaded_model(data.x, data.edge_index)
        
        assert predictions.shape == (num_nodes, 2)

    def test_save_and_load_sage_encoder(
        self,
        sample_node_features: Dict[str, np.ndarray],
        temp_output_dir: Path,
    ) -> None:
        """Test saving and loading GraphSAGE encoder."""
        node_ids = list(sample_node_features.keys())
        features = np.stack([sample_node_features[nid] for nid in node_ids])
        
        # Create encoder
        encoder = InductiveSAGEEncoder(
            input_dim=features.shape[1],
            hidden_dim=16,
            output_dim=8,
            num_layers=2,
            dropout=0.0,
            aggregator='mean',
        )
        
        # Save encoder
        encoder_path = temp_output_dir / "sage_encoder.pt"
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'input_dim': features.shape[1],
            'hidden_dim': 16,
            'output_dim': 8,
            'num_layers': 2,
            'aggregator': 'mean',
        }, encoder_path)
        
        # Verify file exists
        assert encoder_path.exists()
        
        # Load encoder
        checkpoint = torch.load(encoder_path)
        loaded_encoder = InductiveSAGEEncoder(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
            num_layers=checkpoint['num_layers'],
            aggregator=checkpoint['aggregator'],
        )
        loaded_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Verify loaded encoder works
        features_tensor = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            embeddings = loaded_encoder(features_tensor, [])
        
        assert embeddings.shape == (len(node_ids), 8)


class TestTrainingPipelineIntegration:
    """Integration tests for complete training pipelines."""

    def test_features_to_model_pipeline(
        self,
        sample_node_features: Dict[str, np.ndarray],
        sample_edge_list: List[tuple],
        temp_output_dir: Path,
    ) -> None:
        """Test complete pipeline from features to trained model."""
        # Step 1: Prepare features
        node_ids = list(sample_node_features.keys())
        features = np.stack([sample_node_features[nid] for nid in node_ids])
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Step 2: Create graph structure
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        edge_list = []
        for src, dst, _, _ in sample_edge_list:
            if src in node_to_idx and dst in node_to_idx:
                edge_list.append([node_to_idx[src], node_to_idx[dst]])
        
        if len(edge_list) == 0:
            edge_list = [[0, 1], [1, 2], [2, 0]]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Step 3: Create and train model
        model = GCN(
            input_dim=features.shape[1],
            hidden_dim=16,
            output_dim=2,
            dropout=0.5,
        )
        
        # Create dummy labels
        labels = torch.randint(0, 2, (len(node_ids),))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()
        
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            out = model(features_tensor, edge_index)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        
        # Step 4: Save model
        model_path = temp_output_dir / "trained_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': features.shape[1],
            'hidden_dim': 16,
            'output_dim': 2,
            'training_loss': loss.item(),
            'trained_at': datetime.utcnow().isoformat(),
        }, model_path)
        
        # Verify pipeline
        assert model_path.exists()
        checkpoint = torch.load(model_path)
        assert 'training_loss' in checkpoint
        assert 'trained_at' in checkpoint

    def test_incremental_training_workflow(
        self,
        sample_node_features: Dict[str, np.ndarray],
        temp_output_dir: Path,
    ) -> None:
        """Test incremental training with new data."""
        node_ids = list(sample_node_features.keys())
        features = np.stack([sample_node_features[nid] for nid in node_ids])
        
        # Initial training
        model = GCN(
            input_dim=features.shape[1],
            hidden_dim=16,
            output_dim=2,
            dropout=0.5,
        )
        
        edge_index = torch.randint(0, len(node_ids), (2, len(node_ids) * 2))
        labels = torch.randint(0, 2, (len(node_ids),))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()
        
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            out = model(torch.tensor(features, dtype=torch.float32), edge_index)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        
        initial_loss = loss.item()
        
        # Add new data
        new_features = np.random.randn(5, features.shape[1]).astype(np.float32)
        updated_features = np.vstack([features, new_features])
        updated_edge_index = torch.randint(0, len(node_ids) + 5, (2, (len(node_ids) + 5) * 2))
        updated_labels = torch.randint(0, 2, (len(node_ids) + 5,))
        
        # Continue training
        for _ in range(3):
            optimizer.zero_grad()
            out = model(torch.tensor(updated_features, dtype=torch.float32), updated_edge_index)
            loss = criterion(out, updated_labels)
            loss.backward()
            optimizer.step()
        
        # Verify training continued
        assert loss.item() is not None

    def test_model_evaluation_workflow(
        self,
        sample_training_data: tuple,
    ) -> None:
        """Test model evaluation workflow."""
        X, y = sample_training_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create model
        model = GCN(
            input_dim=X.shape[1],
            hidden_dim=16,
            output_dim=2,
            dropout=0.5,
        )
        
        # Train
        edge_index = torch.randint(0, len(X_train), (2, len(X_train) * 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()
        
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            out = model(torch.tensor(X_train, dtype=torch.float32), edge_index)
            loss = criterion(out, torch.tensor(y_train, dtype=torch.long))
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_edge_index = torch.randint(0, len(X_test), (2, len(X_test) * 2))
            predictions = model(torch.tensor(X_test, dtype=torch.float32), test_edge_index)
            predicted_classes = predictions.argmax(dim=1)
            accuracy = (predicted_classes == torch.tensor(y_test)).float().mean()
        
        # Verify evaluation
        assert 0.0 <= accuracy.item() <= 1.0
