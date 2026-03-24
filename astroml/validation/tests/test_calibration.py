"""Tests for calibration curve visualization and analysis."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from astroml.validation.calibration import (
    CalibrationAnalyzer,
    create_sample_fraud_data
)


class TestCalibrationAnalyzer:
    """Test suite for CalibrationAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a calibration analyzer instance."""
        return CalibrationAnalyzer(n_bins=10, strategy='uniform')
    
    @pytest.fixture
    def sample_data(self):
        """Create sample fraud detection data."""
        return create_sample_fraud_data(n_samples=1000, fraud_rate=0.2)
    
    @pytest.fixture
    def perfect_data(self):
        """Create perfectly calibrated data."""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        y_prob = y_true + np.random.normal(0, 0.1, n_samples)
        y_prob = np.clip(y_prob, 0.01, 0.99)
        return y_true, y_prob
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = CalibrationAnalyzer(n_bins=15, strategy='quantile')
        assert analyzer.n_bins == 15
        assert analyzer.strategy == 'quantile'
        assert analyzer.calibration_data == {}
        assert analyzer.metrics == {}
    
    def test_compute_calibration_curve_basic(self, analyzer, sample_data):
        """Test basic calibration curve computation."""
        y_true, y_prob = sample_data
        
        fraction_pos, mean_pred = analyzer.compute_calibration_curve(y_true, y_prob)
        
        assert len(fraction_pos) == len(mean_pred)
        assert len(fraction_pos) <= analyzer.n_bins
        assert all(0 <= fp <= 1 for fp in fraction_pos)
        assert all(0 <= mp <= 1 for mp in mean_pred)
        
        # Check calibration data is stored
        assert 'fraction_of_positives' in analyzer.calibration_data
        assert 'mean_predicted_probability' in analyzer.calibration_data
    
    def test_compute_calibration_curve_length_mismatch(self, analyzer):
        """Test error handling for mismatched input lengths."""
        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.1, 0.8])  # Different length
        
        with pytest.raises(ValueError, match="y_true and y_prob must have the same length"):
            analyzer.compute_calibration_curve(y_true, y_prob)
    
    def test_compute_calibration_curve_invalid_probabilities(self, analyzer):
        """Test error handling for invalid probabilities."""
        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.1, 1.5, -0.1])  # Invalid probabilities
        
        with pytest.raises(ValueError, match="y_prob must be between 0 and 1"):
            analyzer.compute_calibration_curve(y_true, y_prob)
    
    def test_compute_calibration_metrics(self, analyzer, sample_data):
        """Test calibration metrics computation."""
        y_true, y_prob = sample_data
        
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        
        expected_metrics = [
            'brier_score', 'log_loss', 'ece', 'mce', 'ace',
            'overconfidence', 'underconfidence', 'sharpness'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
        
        # Check metric ranges
        assert metrics['brier_score'] >= 0
        assert metrics['log_loss'] >= 0
        assert metrics['ece'] >= 0
        assert metrics['mce'] >= 0
        assert metrics['ace'] >= 0
        assert metrics['sharpness'] >= 0
    
    def test_compute_ece(self, analyzer, sample_data):
        """Test Expected Calibration Error computation."""
        y_true, y_prob = sample_data
        
        # First compute calibration curve
        analyzer.compute_calibration_curve(y_true, y_prob)
        
        # Then compute ECE
        ece = analyzer._compute_ece(y_true, y_prob)
        
        assert isinstance(ece, (int, float))
        assert 0 <= ece <= 1
    
    def test_compute_mce(self, analyzer, sample_data):
        """Test Maximum Calibration Error computation."""
        y_true, y_prob = sample_data
        
        analyzer.compute_calibration_curve(y_true, y_prob)
        mce = analyzer._compute_mce(y_true, y_prob)
        
        assert isinstance(mce, (int, float))
        assert 0 <= mce <= 1
    
    def test_compute_ace(self, analyzer, sample_data):
        """Test Adaptive Calibration Error computation."""
        y_true, y_prob = sample_data
        
        ace = analyzer._compute_ace(y_true, y_prob)
        
        assert isinstance(ace, (int, float))
        assert 0 <= ace <= 1
    
    def test_compute_confidence_metrics(self, analyzer, sample_data):
        """Test overconfidence and underconfidence metrics."""
        y_true, y_prob = sample_data
        
        overconf = analyzer._compute_overconfidence(y_true, y_prob)
        underconf = analyzer._compute_underconfidence(y_true, y_prob)
        
        assert overconf >= 0
        assert underconf >= 0
        # Either overconfidence or underconfidence should be zero (or both)
        assert overconf == 0 or underconf == 0
    
    def test_plot_calibration_curve(self, analyzer, sample_data):
        """Test calibration curve plotting."""
        y_true, y_prob = sample_data
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.plot_calibration_curve(y_true, y_prob, "Test Model")
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # 2x2 subplot layout
        
        # Check that metrics were computed
        assert analyzer.metrics != {}
    
    def test_plot_multiple_models(self, analyzer):
        """Test multi-model calibration comparison."""
        # Create data for multiple models
        models_data = {
            'Model A': create_sample_fraud_data(500, 0.1),
            'Model B': create_sample_fraud_data(500, 0.15),
            'Model C': create_sample_fraud_data(500, 0.2)
        }
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.plot_multiple_models(models_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4
    
    def test_generate_calibration_report(self, analyzer, sample_data):
        """Test calibration report generation."""
        y_true, y_prob = sample_data
        
        report = analyzer.generate_calibration_report(y_true, y_prob, "Test Model")
        
        assert isinstance(report, str)
        assert "Test Model" in report
        assert "Calibration Metrics" in report
        assert "Brier Score" in report
        assert "Expected Calibration Error" in report
        assert "Recommendations" in report
    
    def test_bin_mask_uniform(self, analyzer):
        """Test bin mask generation for uniform strategy."""
        y_prob = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        
        # Test first bin (0-0.1)
        mask = analyzer._get_bin_mask(y_prob, 0)
        expected = y_prob < 0.1
        np.testing.assert_array_equal(mask, expected)
        
        # Test last bin (0.9-1.0)
        mask = analyzer._get_bin_mask(y_prob, 9)
        expected = y_prob >= 0.9
        np.testing.assert_array_equal(mask, expected)
    
    def test_bin_mask_quantile(self):
        """Test bin mask generation for quantile strategy."""
        analyzer = CalibrationAnalyzer(n_bins=5, strategy='quantile')
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # For quantile strategy, bins should have roughly equal samples
        mask = analyzer._get_bin_mask(y_prob, 0)
        assert np.sum(mask) == 2  # First two samples in first quantile
    
    def test_perfect_calibration(self, analyzer, perfect_data):
        """Test metrics on perfectly calibrated data."""
        y_true, y_prob = perfect_data
        
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        
        # Perfect calibration should have low errors
        assert metrics['brier_score'] < 0.2
        assert metrics['ece'] < 0.1
        assert metrics['mce'] < 0.2
    
    def test_edge_cases(self, analyzer):
        """Test edge cases and boundary conditions."""
        # All same prediction
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Should not raise errors
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        assert all(not np.isnan(v) for v in metrics.values())
        
        # All legitimate
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])
        
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        assert all(not np.isnan(v) for v in metrics.values())
        
        # All fraudulent
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.6, 0.7, 0.8, 0.9])
        
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        assert all(not np.isnan(v) for v in metrics.values())


class TestCreateSampleFraudData:
    """Test suite for sample data generation."""
    
    def test_basic_generation(self):
        """Test basic sample data generation."""
        y_true, y_prob = create_sample_fraud_data(n_samples=100, fraud_rate=0.2)
        
        assert len(y_true) == len(y_prob) == 100
        assert all(y in [0, 1] for y in y_true)
        assert all(0 <= p <= 1 for p in y_prob)
        assert abs(np.mean(y_true) - 0.2) < 0.05  # Within expected range
    
    def test_different_parameters(self):
        """Test with different parameters."""
        y_true, y_prob = create_sample_fraud_data(n_samples=50, fraud_rate=0.5)
        
        assert len(y_true) == 50
        assert abs(np.mean(y_true) - 0.5) < 0.1
    
    def test_reproducibility(self):
        """Test that data generation is reproducible."""
        y_true1, y_prob1 = create_sample_fraud_data()
        y_true2, y_prob2 = create_sample_fraud_data()
        
        np.testing.assert_array_equal(y_true1, y_true2)
        np.testing.assert_array_equal(y_prob1, y_prob2)


class TestIntegration:
    """Integration tests for the calibration module."""
    
    def test_full_workflow(self):
        """Test complete calibration analysis workflow."""
        # Create sample data
        y_true, y_prob = create_sample_fraud_data(n_samples=1000)
        
        # Initialize analyzer
        analyzer = CalibrationAnalyzer(n_bins=8, strategy='quantile')
        
        # Compute calibration curve
        fraction_pos, mean_pred = analyzer.compute_calibration_curve(y_true, y_prob)
        
        # Compute metrics
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        
        # Generate plot
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.plot_calibration_curve(y_true, y_prob, "Integration Test")
        
        # Generate report
        report = analyzer.generate_calibration_report(y_true, y_prob, "Integration Test")
        
        # Verify all components work together
        assert len(fraction_pos) > 0
        assert len(metrics) == 8
        assert isinstance(fig, plt.Figure)
        assert len(report) > 100
        assert "Integration Test" in report
    
    def test_multiple_models_comparison(self):
        """Test multi-model comparison workflow."""
        # Generate different quality models
        models_data = {
            'Poor Model': create_sample_fraud_data(200, 0.1),
            'Good Model': create_sample_fraud_data(200, 0.2),
            'Excellent Model': create_sample_fraud_data(200, 0.15)
        }
        
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        with patch('matplotlib.pyplot.show'):
            fig = analyzer.plot_multiple_models(models_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4


if __name__ == "__main__":
    pytest.main([__file__])
