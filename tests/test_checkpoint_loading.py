"""Tests for model checkpoint loading error handling.

This module tests that checkpoint loading properly handles errors and
does not fail silently, addressing the issue of silent failures.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np


class TestDeepSVDDCheckpointLoading:
    """Tests for DeepSVDD checkpoint loading error handling."""

    def test_load_checkpoint_missing_file_raises_error(self):
        """Test that loading a non-existent checkpoint raises FileNotFoundError."""
        from astroml.models.deep_svdd_trainer import DeepSVDDTrainer
        from astroml.models.deep_svdd import DeepSVDD

        model = DeepSVDD(input_dim=10, hidden_dims=[8, 4], device='cpu')
        trainer = DeepSVDDTrainer(model, device='cpu')

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            trainer.load_checkpoint('nonexistent_checkpoint.pth')

    def test_load_checkpoint_missing_required_key_raises_error(self):
        """Test that loading a checkpoint missing required keys raises ValueError."""
        from astroml.models.deep_svdd_trainer import DeepSVDDTrainer
        from astroml.models.deep_svdd import DeepSVDD

        model = DeepSVDD(input_dim=10, hidden_dims=[8, 4], device='cpu')
        trainer = DeepSVDDTrainer(model, device='cpu')

        # Create a checkpoint missing the 'center' key
        incomplete_checkpoint = {
            'model_state_dict': model.state_dict(),
            # Missing 'center' key
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(incomplete_checkpoint, f.name)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Checkpoint missing required key: center"):
                trainer.load_checkpoint(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_corrupted_file_raises_error(self):
        """Test that loading a corrupted checkpoint raises ValueError."""
        from astroml.models.deep_svdd_trainer import DeepSVDDTrainer
        from astroml.models.deep_svdd import DeepSVDD

        model = DeepSVDD(input_dim=10, hidden_dims=[8, 4], device='cpu')
        trainer = DeepSVDDTrainer(model, device='cpu')

        # Create a file with invalid content
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False, mode='w') as f:
            f.write("corrupted data that is not a valid checkpoint")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load checkpoint"):
                trainer.load_checkpoint(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_state_dict_mismatch_raises_error(self):
        """Test that loading a checkpoint with mismatched state dict raises RuntimeError."""
        from astroml.models.deep_svdd_trainer import DeepSVDDTrainer
        from astroml.models.deep_svdd import DeepSVDD

        model = DeepSVDD(input_dim=10, hidden_dims=[8, 4], device='cpu')
        trainer = DeepSVDDTrainer(model, device='cpu')

        # Create a checkpoint with a different model's state dict
        different_model = DeepSVDD(input_dim=20, hidden_dims=[16, 8], device='cpu')
        checkpoint = {
            'model_state_dict': different_model.state_dict(),
            'center': torch.zeros(10),
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError, match="State dict does not match model architecture"):
                trainer.load_checkpoint(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_valid_checkpoint_returns_true(self):
        """Test that loading a valid checkpoint returns True."""
        from astroml.models.deep_svdd_trainer import DeepSVDDTrainer
        from astroml.models.deep_svdd import DeepSVDD

        model = DeepSVDD(input_dim=10, hidden_dims=[8, 4], device='cpu')
        trainer = DeepSVDDTrainer(model, device='cpu')

        # Create a valid checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'center': torch.zeros(10),
            'scaler': None,
            'training_history': {'train_loss': [1.0, 0.5]},
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_path = f.name

        try:
            result = trainer.load_checkpoint(temp_path)
            assert result is True
            assert trainer.training_history == {'train_loss': [1.0, 0.5]}
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_uses_weights_only(self):
        """Test that checkpoint loading uses weights_only=True for security."""
        from astroml.models.deep_svdd_trainer import DeepSVDDTrainer
        from astroml.models.deep_svdd import DeepSVDD

        model = DeepSVDD(input_dim=10, hidden_dims=[8, 4], device='cpu')
        trainer = DeepSVDDTrainer(model, device='cpu')

        # Create a valid checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'center': torch.zeros(10),
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_path = f.name

        try:
            # Mock torch.load to verify weights_only parameter
            with patch('torch.load') as mock_load:
                mock_load.return_value = checkpoint
                trainer.load_checkpoint(temp_path)
                # Verify that weights_only=True was passed
                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args[1]
                assert call_kwargs.get('weights_only') is True
        finally:
            os.unlink(temp_path)


class TestTemporalCheckpointLoading:
    """Tests for Temporal model checkpoint loading error handling."""

    def test_load_checkpoint_missing_file_raises_error(self):
        """Test that loading a non-existent checkpoint raises FileNotFoundError."""
        from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig

        config = TemporalTrainingConfig(input_dim=10, epochs=1)
        trainer = TemporalTrainer(config)

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            trainer.load_checkpoint('nonexistent_checkpoint.pth')

    def test_load_checkpoint_missing_required_key_raises_error(self):
        """Test that loading a checkpoint missing required keys raises ValueError."""
        from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig

        config = TemporalTrainingConfig(input_dim=10, epochs=1)
        trainer = TemporalTrainer(config)

        # Create a checkpoint missing the 'optimizer_state_dict' key
        incomplete_checkpoint = {
            'model_state_dict': trainer.model.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'training_history': {},
            # Missing 'optimizer_state_dict' key
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(incomplete_checkpoint, f.name)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Checkpoint missing required key: optimizer_state_dict"):
                trainer.load_checkpoint(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_corrupted_file_raises_error(self):
        """Test that loading a corrupted checkpoint raises ValueError."""
        from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig

        config = TemporalTrainingConfig(input_dim=10, epochs=1)
        trainer = TemporalTrainer(config)

        # Create a file with invalid content
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False, mode='w') as f:
            f.write("corrupted data that is not a valid checkpoint")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load checkpoint"):
                trainer.load_checkpoint(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_state_dict_mismatch_raises_error(self):
        """Test that loading a checkpoint with mismatched state dict raises RuntimeError."""
        from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig
        from astroml.models.temporal import TemporalGCN

        config = TemporalTrainingConfig(input_dim=10, epochs=1)
        trainer = TemporalTrainer(config)

        # Create a checkpoint with a different model's state dict
        different_config = TemporalTrainingConfig(input_dim=20, epochs=1)
        different_trainer = TemporalTrainer(different_config)
        checkpoint = {
            'model_state_dict': different_trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'training_history': {},
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError, match="Model state dict does not match architecture"):
                trainer.load_checkpoint(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_valid_checkpoint_returns_true(self):
        """Test that loading a valid checkpoint returns True."""
        from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig

        config = TemporalTrainingConfig(input_dim=10, epochs=1)
        trainer = TemporalTrainer(config)

        # Create a valid checkpoint
        checkpoint = {
            'epoch': 5,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'training_history': {'train_loss': [1.0, 0.5]},
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_path = f.name

        try:
            result = trainer.load_checkpoint(temp_path)
            assert result is True
            assert trainer.training_history == {'train_loss': [1.0, 0.5]}
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_uses_weights_only(self):
        """Test that checkpoint loading uses weights_only=True for security."""
        from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig

        config = TemporalTrainingConfig(input_dim=10, epochs=1)
        trainer = TemporalTrainer(config)

        # Create a valid checkpoint
        checkpoint = {
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'training_history': {},
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_path = f.name

        try:
            # Mock torch.load to verify weights_only parameter
            with patch('torch.load') as mock_load:
                mock_load.return_value = checkpoint
                trainer.load_checkpoint(temp_path)
                # Verify that weights_only=True was passed
                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args[1]
                assert call_kwargs.get('weights_only') is True
        finally:
            os.unlink(temp_path)

    def test_load_checkpoint_missing_epoch_logs_warning(self):
        """Test that loading checkpoint without epoch info logs appropriately."""
        from astroml.training.temporal import TemporalTrainer, TemporalTrainingConfig

        config = TemporalTrainingConfig(input_dim=10, epochs=1)
        trainer = TemporalTrainer(config)

        # Create a checkpoint without epoch info
        checkpoint = {
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'training_history': {},
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(checkpoint, f.name)
            temp_path = f.name

        try:
            result = trainer.load_checkpoint(temp_path)
            assert result is True
        finally:
            os.unlink(temp_path)
