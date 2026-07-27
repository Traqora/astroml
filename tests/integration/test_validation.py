"""Integration tests for validation and calibration pipeline.

These tests verify the complete workflow from model predictions
to validation, calibration, and quality assurance.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from astroml.validation.calibration import CalibrationAnalyzer
from astroml.validation.data_quality import (
    DataQualityReport,
    TemporalValidator,
    ValidationResult,
)
from astroml.validation.validator import (
    TransactionValidator,
    validate_transaction,
    CorruptionType,
)


class TestCalibrationIntegration:
    """Integration tests for model calibration."""

    def test_calibration_analysis_workflow(
        self,
        fraud_labels: np.ndarray,
        fraud_scores: np.ndarray,
    ) -> None:
        """Test complete calibration analysis workflow."""
        analyzer = CalibrationAnalyzer(n_bins=10, strategy='uniform')
        
        # Compute calibration curve
        fraction_positives, mean_predicted = analyzer.compute_calibration_curve(
            fraud_labels, fraud_scores
        )
        
        # Verify calibration data
        assert len(fraction_positives) == len(mean_predicted)
        assert len(fraction_positives) <= 10
        assert np.all(fraction_positives >= 0)
        assert np.all(fraction_positives <= 1)
        assert np.all(mean_predicted >= 0)
        assert np.all(mean_predicted <= 1)

    def test_calibration_metrics_computation(
        self,
        fraud_labels: np.ndarray,
        fraud_scores: np.ndarray,
    ) -> None:
        """Test comprehensive calibration metrics computation."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        # Compute metrics
        metrics = analyzer.compute_calibration_metrics(
            fraud_labels, fraud_scores
        )
        
        # Verify metrics
        assert 'brier_score' in metrics
        assert 'log_loss' in metrics
        assert metrics['brier_score'] >= 0
        assert metrics['log_loss'] >= 0

    def test_calibration_with_perfect_predictions(
        self,
    ) -> None:
        """Test calibration with perfectly calibrated predictions."""
        # Create perfectly calibrated data
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = y_true.astype(float) + np.random.normal(0, 0.05, n_samples)
        y_prob = np.clip(y_prob, 0.01, 0.99)
        
        analyzer = CalibrationAnalyzer(n_bins=10)
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        
        # Perfect calibration should have low Brier score
        assert metrics['brier_score'] < 0.1

    def test_calibration_with_random_predictions(
        self,
    ) -> None:
        """Test calibration with random (uncalibrated) predictions."""
        # Create random predictions
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.uniform(0, 1, n_samples)
        
        analyzer = CalibrationAnalyzer(n_bins=10)
        metrics = analyzer.compute_calibration_metrics(y_true, y_prob)
        
        # Random predictions should have higher Brier score
        assert metrics['brier_score'] >= 0.2


class TestDataQualityIntegration:
    """Integration tests for data quality validation."""

    def test_transaction_validation_workflow(
        self,
        sample_transaction_data: List[Dict[str, Any]],
    ) -> None:
        """Test complete transaction validation workflow."""
        validator = TransactionValidator(
            required_fields={"hash", "source_account", "created_at", "fee"},
            field_types={"fee": int, "operation_count": int},
        )
        
        # Validate transactions
        results = validator.validate_batch(sample_transaction_data)
        
        # Verify results
        assert len(results) == len(sample_transaction_data)
        assert all(isinstance(r, type(results[0])) for r in results)

    def test_data_quality_report_generation(
        self,
    ) -> None:
        """Test comprehensive data quality report generation."""
        # Create sample transactions with various issues
        transactions = [
            {"id": "tx1", "source_account": "GAAA", "amount": 100.0},
            {"id": "tx2", "amount": 50.0},  # Missing source_account
            {"id": "tx3", "source_account": "GBBB", "amount": "invalid"},  # Invalid type
            {"id": "tx4", "source_account": "GCCC", "amount": 200.0},
        ]
        
        validator = TransactionValidator(
            required_fields={"id", "source_account", "amount"},
            field_types={"amount": (int, float)},
        )
        
        # Validate and generate report
        results = validator.validate_batch(transactions)
        
        valid_count = sum(1 for r in results if r.is_valid)
        report = DataQualityReport(
            total_records=len(transactions),
            valid_records=valid_count,
            validation_results=[
                ValidationResult(
                    is_valid=r.is_valid,
                    error_type=r.errors[0].error_type if r.errors else None,
                    message=r.errors[0].message if r.errors else "Valid",
                )
                for r in results
            ],
        )
        
        # Verify report
        assert report.total_records == 4
        assert report.valid_records == 2
        assert report.quality_score == 50.0
        assert len(report.error_types) > 0

    def test_temporal_validation_workflow(
        self,
    ) -> None:
        """Test temporal data validation workflow."""
        validator = TemporalValidator(timestamp_field="timestamp")
        
        # Create transactions with timestamps
        base_time = datetime(2024, 1, 1)
        transactions = [
            {"id": "tx1", "timestamp": base_time},
            {"id": "tx2", "timestamp": base_time + timedelta(hours=1)},
            {"id": "tx3", "timestamp": base_time + timedelta(hours=2)},
        ]
        
        # Validate ordering
        result = validator.validate_timestamp_ordering(transactions)
        
        # Should be valid (monotonically increasing)
        assert result.is_valid

    def test_temporal_validation_with_out_of_order(
        self,
    ) -> None:
        """Test temporal validation with out-of-order timestamps."""
        validator = TemporalValidator(timestamp_field="timestamp")
        
        # Create transactions with out-of-order timestamps
        base_time = datetime(2024, 1, 1)
        transactions = [
            {"id": "tx1", "timestamp": base_time + timedelta(hours=2)},
            {"id": "tx2", "timestamp": base_time},
            {"id": "tx3", "timestamp": base_time + timedelta(hours=1)},
        ]
        
        # Validate ordering
        result = validator.validate_timestamp_ordering(transactions)
        
        # Should be invalid
        assert not result.is_valid


class TestValidationPipelineIntegration:
    """Integration tests for complete validation pipeline."""

    def test_model_prediction_validation_workflow(
        self,
        fraud_labels: np.ndarray,
        fraud_scores: np.ndarray,
    ) -> None:
        """Test validation of model predictions before calibration."""
        # Validate prediction format
        assert len(fraud_labels) == len(fraud_scores)
        assert np.all((fraud_scores >= 0) & (fraud_scores <= 1))
        
        # Check for NaN or infinite values
        assert not np.any(np.isnan(fraud_scores))
        assert not np.any(np.isinf(fraud_scores))
        
        # Proceed with calibration
        analyzer = CalibrationAnalyzer(n_bins=10)
        metrics = analyzer.compute_calibration_metrics(fraud_labels, fraud_scores)
        
        # Verify metrics are valid
        assert all(np.isfinite(v) for v in metrics.values())

    def test_end_to_end_validation_pipeline(
        self,
        sample_transaction_data: List[Dict[str, Any]],
        fraud_labels: np.ndarray,
        fraud_scores: np.ndarray,
    ) -> None:
        """Test complete validation pipeline from transactions to calibrated metrics."""
        # Step 1: Validate transaction data
        validator = TransactionValidator(
            required_fields={"hash", "source_account", "created_at"},
        )
        tx_results = validator.validate_batch(sample_transaction_data)
        
        # Step 2: Filter valid transactions
        valid_tx_count = sum(1 for r in tx_results if r.is_valid)
        assert valid_tx_count > 0
        
        # Step 3: Validate prediction data
        assert len(fraud_labels) == len(fraud_scores)
        assert not np.any(np.isnan(fraud_scores))
        
        # Step 4: Compute calibration metrics
        analyzer = CalibrationAnalyzer(n_bins=10)
        metrics = analyzer.compute_calibration_metrics(fraud_labels, fraud_scores)
        
        # Step 5: Verify pipeline results
        assert 'brier_score' in metrics
        assert metrics['brier_score'] >= 0
        assert valid_tx_count == len(sample_transaction_data)

    def test_validation_with_corrupted_data(
        self,
    ) -> None:
        """Test validation pipeline with corrupted data."""
        # Create corrupted transactions
        corrupted_transactions = [
            {"id": None, "source_account": "GAAA", "amount": 100.0},  # Null ID
            {"id": "tx2", "amount": 50.0},  # Missing source_account
            {"amount": 200.0},  # Missing both id and source_account
        ]
        
        validator = TransactionValidator(
            required_fields={"id", "source_account"},
        )
        
        # Validate
        results = validator.validate_batch(corrupted_transactions)
        
        # All should be invalid
        assert all(not r.is_valid for r in results)
        
        # Check error types
        error_types = {r.errors[0].error_type for r in results if r.errors}
        assert CorruptionType.MISSING_FIELD in error_types

    def test_validation_report_persistence(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test saving and loading validation reports."""
        # Create a validation report
        report = DataQualityReport(
            total_records=100,
            valid_records=95,
            validation_results=[
                ValidationResult(
                    is_valid=True,
                    message="Valid transaction",
                )
                for _ in range(95)
            ] + [
                ValidationResult(
                    is_valid=False,
                    error_type="MISSING_FIELD",
                    message="Missing required field",
                )
                for _ in range(5)
            ],
        )
        
        # Save report
        report_path = temp_output_dir / "validation_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump({
                'total_records': report.total_records,
                'valid_records': report.valid_records,
                'quality_score': report.quality_score,
                'error_types': list(report.error_types),
            }, f)
        
        # Verify file exists
        assert report_path.exists()
        
        # Load and verify
        with open(report_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['total_records'] == 100
        assert loaded['valid_records'] == 95
        assert loaded['quality_score'] == 95.0


class TestCalibrationVisualizationIntegration:
    """Integration tests for calibration visualization."""

    def test_calibration_plot_generation(
        self,
        fraud_labels: np.ndarray,
        fraud_scores: np.ndarray,
        temp_output_dir: Path,
    ) -> None:
        """Test calibration plot generation and saving."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        # Compute calibration curve
        fraction_positives, mean_predicted = analyzer.compute_calibration_curve(
            fraud_labels, fraud_scores
        )
        
        # Generate plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(mean_predicted, fraction_positives, 's-', label='Model')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve')
        plt.legend()
        
        # Save plot
        plot_path = temp_output_dir / "calibration_curve.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Verify file exists
        assert plot_path.exists()

    def test_calibration_metrics_report(
        self,
        fraud_labels: np.ndarray,
        fraud_scores: np.ndarray,
        temp_output_dir: Path,
    ) -> None:
        """Test generating comprehensive calibration metrics report."""
        analyzer = CalibrationAnalyzer(n_bins=10)
        
        # Compute metrics
        metrics = analyzer.compute_calibration_metrics(fraud_labels, fraud_scores)
        
        # Generate report
        report = {
            'calibration_metrics': metrics,
            'n_samples': len(fraud_labels),
            'n_bins': analyzer.n_bins,
            'strategy': analyzer.strategy,
            'generated_at': datetime.utcnow().isoformat(),
        }
        
        # Save report
        report_path = temp_output_dir / "calibration_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Verify file exists and contains expected data
        assert report_path.exists()
        with open(report_path, 'r') as f:
            loaded = json.load(f)
        
        assert 'calibration_metrics' in loaded
        assert 'brier_score' in loaded['calibration_metrics']
        assert loaded['n_samples'] == len(fraud_labels)
