"""Extended data quality validation tests.

Tests for the new data quality validation utilities covering temporal consistency,
referential integrity, business rules, and statistical validation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from astroml.validation.data_quality import (
    BusinessRulesValidator,
    DataQualityReport,
    DataQualityValidator,
    ReferentialIntegrityValidator,
    StatisticalValidator,
    TemporalValidator,
    ValidationResult,
    check_referential_integrity,
    check_temporal_consistency,
    validate_data_quality,
)


class TestTemporalValidator:
    """Test temporal data quality validation."""

    def test_timestamp_ordering_valid(self):
        """Test valid timestamp ordering."""
        validator = TemporalValidator()
        base_time = datetime.utcnow()
        
        transactions = [
            {"id": f"tx_{i}", "timestamp": (base_time + timedelta(hours=i)).isoformat()}
            for i in range(5)
        ]
        
        result = validator.validate_timestamp_ordering(transactions)
        assert result.is_valid
        assert result.message == "Timestamps are properly ordered"

    def test_timestamp_ordering_invalid(self):
        """Test invalid timestamp ordering."""
        validator = TemporalValidator()
        base_time = datetime.utcnow()
        
        transactions = [
            {"id": "tx_0", "timestamp": (base_time + timedelta(hours=0)).isoformat()},
            {"id": "tx_1", "timestamp": (base_time + timedelta(hours=2)).isoformat()},
            {"id": "tx_2", "timestamp": (base_time + timedelta(hours=1)).isoformat()},  # Out of order
        ]
        
        result = validator.validate_timestamp_ordering(transactions)
        assert not result.is_valid
        assert result.error_type == "TIMESTAMP_ORDER_VIOLATION"
        assert "index 1" in result.message

    def test_timestamp_ordering_missing_field(self):
        """Test missing timestamp field."""
        validator = TemporalValidator()
        
        transactions = [
            {"id": "tx_1", "amount": 100},  # Missing timestamp
            {"id": "tx_2", "timestamp": datetime.utcnow().isoformat()},
        ]
        
        result = validator.validate_timestamp_ordering(transactions)
        assert not result.is_valid
        assert result.error_type == "MISSING_TIMESTAMP"

    def test_timestamp_ordering_invalid_format(self):
        """Test invalid timestamp format."""
        validator = TemporalValidator()
        
        transactions = [
            {"id": "tx_1", "timestamp": "not-a-timestamp"},
        ]
        
        result = validator.validate_timestamp_ordering(transactions)
        assert not result.is_valid
        assert result.error_type == "INVALID_TIMESTAMP_FORMAT"

    def test_future_timestamps_valid(self):
        """Test valid future timestamp detection (no future timestamps)."""
        validator = TemporalValidator()
        past_time = datetime.utcnow() - timedelta(hours=1)
        
        transactions = [
            {"id": "tx_1", "timestamp": past_time.isoformat()},
        ]
        
        result = validator.validate_future_timestamps(transactions)
        assert result.is_valid
        assert result.message == "No future timestamps detected"

    def test_future_timestamps_detected(self):
        """Test detection of future timestamps."""
        validator = TemporalValidator()
        future_time = datetime.utcnow() + timedelta(hours=1)
        
        transactions = [
            {"id": "tx_1", "timestamp": future_time.isoformat()},
        ]
        
        result = validator.validate_future_timestamps(transactions)
        assert not result.is_valid
        assert result.error_type == "FUTURE_TIMESTAMP"
        assert "future timestamps" in result.message

    def test_future_timestamps_with_tolerance(self):
        """Test future timestamp detection with tolerance."""
        validator = TemporalValidator()
        near_future = datetime.utcnow() + timedelta(minutes=2)  # Within 5-minute tolerance
        far_future = datetime.utcnow() + timedelta(hours=1)  # Beyond tolerance
        
        transactions = [
            {"id": "tx_near", "timestamp": near_future.isoformat()},
            {"id": "tx_far", "timestamp": far_future.isoformat()},
        ]
        
        result = validator.validate_future_timestamps(transactions, tolerance_minutes=5)
        assert not result.is_valid
        assert len(result.details["future_transactions"]) == 1
        assert result.details["future_transactions"][0]["id"] == "tx_far"

    def test_empty_transaction_list(self):
        """Test validation with empty transaction list."""
        validator = TemporalValidator()
        
        result = validator.validate_timestamp_ordering([])
        assert result.is_valid
        assert result.message == "Empty transaction list"
        
        result = validator.validate_future_timestamps([])
        assert result.is_valid
        assert result.message == "Empty transaction list"


class TestReferentialIntegrityValidator:
    """Test referential integrity validation."""

    def test_valid_account_format(self):
        """Test valid Stellar account formats."""
        validator = ReferentialIntegrityValidator()
        
        valid_accounts = [
            "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "GABCDEFGHIJKLMN0PQRSTUVWXYZ0123456789012345"
        ]
        
        for account in valid_accounts:
            result = validator.validate_account_format(account)
            assert result.is_valid
            assert result.message == "Account format is valid"

    def test_invalid_account_format(self):
        """Test invalid Stellar account formats."""
        validator = ReferentialIntegrityValidator()
        
        invalid_accounts = [
            "XABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",  # Wrong prefix
            "GABCD123",  # Too short
            "gabcd1234567890abcdefghijklmnopqrstuvwxyz1234567890",  # Lowercase
            "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#",  # Extra chars
            12345,  # Not a string
        ]
        
        for account in invalid_accounts:
            result = validator.validate_account_format(account)
            assert not result.is_valid
            assert result.error_type in ["INVALID_ACCOUNT_FORMAT", "INVALID_ACCOUNT_TYPE"]

    def test_valid_asset_format(self):
        """Test valid asset code formats."""
        validator = ReferentialIntegrityValidator()
        
        valid_codes = ["XLM", "USD", "BTC", "CUSTOM123", "ASSETCODE"]
        
        for code in valid_codes:
            result = validator.validate_asset_format(code)
            assert result.is_valid
            assert result.message == "Asset code format is valid"

    def test_invalid_asset_format(self):
        """Test invalid asset code formats."""
        validator = ReferentialIntegrityValidator()
        
        invalid_codes = [
            "xlm",  # Lowercase
            "",  # Empty
            "TOOLONGASSETCODE123",  # Too long
            "asset-with-dash",  # Invalid characters
            123,  # Not a string
        ]
        
        for code in invalid_codes:
            result = validator.validate_asset_format(code)
            assert not result.is_valid
            assert result.error_type in ["INVALID_ASSET_FORMAT", "INVALID_ASSET_TYPE"]

    def test_valid_ledger_sequence(self):
        """Test valid ledger sequences."""
        validator = ReferentialIntegrityValidator()
        
        valid_sequences = [1, 100, 12345, 999999]
        
        for seq in valid_sequences:
            result = validator.validate_ledger_sequence(seq)
            assert result.is_valid
            assert result.message == "Ledger sequence is valid"

    def test_invalid_ledger_sequence(self):
        """Test invalid ledger sequences."""
        validator = ReferentialIntegrityValidator()
        
        invalid_sequences = [0, -1, -100, "123", 123.45]
        
        for seq in invalid_sequences:
            result = validator.validate_ledger_sequence(seq)
            assert not result.is_valid
            assert result.error_type in ["INVALID_LEDGER_SEQUENCE", "INVALID_LEDGER_SEQUENCE_TYPE"]


class TestBusinessRulesValidator:
    """Test business rules validation."""

    def test_valid_fee(self):
        """Test valid fee values."""
        validator = BusinessRulesValidator()
        
        valid_fees = [0, 100, 1000, 50000]
        
        for fee in valid_fees:
            result = validator.validate_fee_non_negative(fee)
            assert result.is_valid
            assert result.message == "Fee is valid"

    def test_invalid_fee(self):
        """Test invalid fee values."""
        validator = BusinessRulesValidator()
        
        invalid_fees = [-1, -100, -0.1]
        
        for fee in invalid_fees:
            result = validator.validate_fee_non_negative(fee)
            assert not result.is_valid
            assert result.error_type == "NEGATIVE_FEE"

    def test_invalid_fee_type(self):
        """Test invalid fee types."""
        validator = BusinessRulesValidator()
        
        invalid_types = ["100", "free", None, []]
        
        for fee in invalid_types:
            result = validator.validate_fee_non_negative(fee)
            assert not result.is_valid
            assert result.error_type == "INVALID_FEE_TYPE"

    def test_valid_amount(self):
        """Test valid amount values."""
        validator = BusinessRulesValidator()
        
        valid_amounts = [0, 0.1, 100.0, 1000000.5]
        
        for amount in valid_amounts:
            result = validator.validate_amount_non_negative(amount)
            assert result.is_valid
            assert result.message == "Amount is valid"

    def test_invalid_amount(self):
        """Test invalid amount values."""
        validator = BusinessRulesValidator()
        
        invalid_amounts = [-0.1, -100.0]
        
        for amount in invalid_amounts:
            result = validator.validate_amount_non_negative(amount)
            assert not result.is_valid
            assert result.error_type == "NEGATIVE_AMOUNT"

    def test_valid_operation_count(self):
        """Test valid operation counts."""
        validator = BusinessRulesValidator()
        
        valid_counts = [1, 10, 50, 100]
        
        for count in valid_counts:
            result = validator.validate_operation_count(count)
            assert result.is_valid
            assert result.message == "Operation count is valid"

    def test_invalid_operation_count(self):
        """Test invalid operation counts."""
        validator = BusinessRulesValidator()
        
        invalid_counts = [0, -1, 101, 1000]
        
        for count in invalid_counts:
            result = validator.validate_operation_count(count)
            assert not result.is_valid
            assert result.error_type == "INVALID_OPERATION_COUNT"

    def test_valid_balance(self):
        """Test valid balance values."""
        validator = BusinessRulesValidator()
        
        valid_balances = [0, 0.1, 100.0, 1000000.123456789]
        
        for balance in valid_balances:
            result = validator.validate_balance_format(balance)
            assert result.is_valid
            assert result.message == "Balance format is valid"

    def test_none_balance(self):
        """Test None balance (should be valid)."""
        validator = BusinessRulesValidator()
        
        result = validator.validate_balance_format(None)
        assert result.is_valid
        assert result.message == "Balance can be None"

    def test_invalid_balance(self):
        """Test invalid balance values."""
        validator = BusinessRulesValidator()
        
        invalid_balances = [float('inf'), float('-inf'), float('nan'), "100", None]
        
        for balance in invalid_balances:
            if balance is None:
                continue  # None is valid
            result = validator.validate_balance_format(balance)
            assert not result.is_valid
            assert result.error_type in ["INVALID_BALANCE_TYPE", "INVALID_BALANCE_VALUE"]


class TestStatisticalValidator:
    """Test statistical validation."""

    def test_no_outliers(self):
        """Test outlier detection with no outliers."""
        validator = StatisticalValidator()
        
        amounts = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        
        result = validator.detect_amount_outliers(amounts)
        assert result.is_valid
        assert result.message == "No amount outliers detected"
        assert "q1" in result.details
        assert "q3" in result.details

    def test_outliers_detected(self):
        """Test outlier detection with outliers."""
        validator = StatisticalValidator()
        
        amounts = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 10000.0, 0.0001]
        
        result = validator.detect_amount_outliers(amounts)
        assert not result.is_valid
        assert result.error_type == "AMOUNT_OUTLIERS_DETECTED"
        assert len(result.details["outliers"]) > 0
        assert "lower_bound" in result.details
        assert "upper_bound" in result.details

    def test_insufficient_data_for_outliers(self):
        """Test outlier detection with insufficient data."""
        validator = StatisticalValidator()
        
        amounts = [10.0, 15.0]  # Too few values
        
        result = validator.detect_amount_outliers(amounts)
        assert result.is_valid
        assert "Insufficient data" in result.message

    def test_no_timestamp_gaps(self):
        """Test timestamp gap detection with no unusual gaps."""
        validator = StatisticalValidator()
        base_time = datetime.utcnow()
        
        timestamps = [base_time + timedelta(minutes=5*i) for i in range(10)]
        
        result = validator.detect_timestamp_gaps(timestamps, gap_threshold_minutes=60)
        assert result.is_valid
        assert result.message == "No unusual timestamp gaps detected"

    def test_timestamp_gaps_detected(self):
        """Test timestamp gap detection with unusual gaps."""
        validator = StatisticalValidator()
        base_time = datetime.utcnow()
        
        timestamps = [base_time + timedelta(minutes=5*i) for i in range(5)]
        timestamps.append(base_time + timedelta(days=1))  # Large gap
        
        result = validator.detect_timestamp_gaps(timestamps, gap_threshold_minutes=60)
        assert not result.is_valid
        assert result.error_type == "UNUSUAL_TIMESTAMP_GAPS"
        assert len(result.details["unusual_gaps"]) > 0

    def test_insufficient_timestamps_for_gaps(self):
        """Test gap detection with insufficient timestamps."""
        validator = StatisticalValidator()
        
        timestamps = [datetime.utcnow()]
        
        result = validator.detect_timestamp_gaps(timestamps)
        assert result.is_valid
        assert "Insufficient timestamps" in result.message

    def test_no_duplicate_patterns(self):
        """Test duplicate pattern detection with no duplicates."""
        validator = StatisticalValidator()
        
        transactions = [
            {"id": "tx_1", "amount": 100.0, "source_account": "ACC1"},
            {"id": "tx_2", "amount": 200.0, "source_account": "ACC2"},
            {"id": "tx_3", "amount": 300.0, "source_account": "ACC3"},
        ]
        
        result = validator.detect_duplicate_patterns(transactions, ["amount", "source_account"])
        assert result.is_valid
        assert result.message == "No duplicate patterns detected"

    def test_duplicate_patterns_detected(self):
        """Test duplicate pattern detection with duplicates."""
        validator = StatisticalValidator()
        
        transactions = [
            {"id": "tx_1", "amount": 100.0, "source_account": "ACC1"},
            {"id": "tx_2", "amount": 100.0, "source_account": "ACC1"},  # Duplicate pattern
            {"id": "tx_3", "amount": 200.0, "source_account": "ACC2"},
        ]
        
        result = validator.detect_duplicate_patterns(transactions, ["amount", "source_account"])
        assert not result.is_valid
        assert result.error_type == "DUPLICATE_PATTERNS_DETECTED"
        assert len(result.details["repeated_patterns"]) > 0

    def test_empty_transactions_for_patterns(self):
        """Test pattern detection with empty transactions."""
        validator = StatisticalValidator()
        
        result = validator.detect_duplicate_patterns([], ["amount"])
        assert result.is_valid
        assert "No transactions" in result.message


class TestDataQualityValidator:
    """Test comprehensive data quality validator."""

    def test_comprehensive_validation_valid(self):
        """Test comprehensive validation with valid data."""
        validator = DataQualityValidator()
        base_time = datetime.utcnow()
        
        transactions = [
            {
                "id": "tx_1",
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "source_account": "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
                "asset_code": "XLM",
                "ledger_sequence": 100 + i,
                "fee": 100,
                "amount": 100.0,
                "operation_count": 1
            }
            for i in range(3)
        ]
        
        report = validator.validate_batch(transactions)
        assert isinstance(report, DataQualityReport)
        assert report.total_records == 3
        assert len(report.validation_results) > 0

    def test_comprehensive_validation_invalid(self):
        """Test comprehensive validation with invalid data."""
        validator = DataQualityValidator()
        
        transactions = [
            {
                "id": "tx_1",
                "source_account": "INVALID_ACCOUNT",  # Invalid format
                "asset_code": "invalid_asset",  # Invalid format
                "ledger_sequence": -1,  # Invalid
                "fee": -100,  # Invalid
                "amount": -50.0,  # Invalid
                "operation_count": 0,  # Invalid
            }
        ]
        
        report = validator.validate_batch(transactions)
        assert isinstance(report, DataQualityReport)
        assert report.total_records == 1
        assert len(report.validation_results) > 0
        
        # Check that errors were detected
        error_results = [r for r in report.validation_results if not r.is_valid]
        assert len(error_results) > 0

    def test_empty_batch_validation(self):
        """Test validation with empty batch."""
        validator = DataQualityValidator()
        
        report = validator.validate_batch([])
        assert isinstance(report, DataQualityReport)
        assert report.total_records == 0
        assert report.valid_records == 0
        assert report.quality_score == 0.0

    def test_report_quality_score(self):
        """Test data quality report score calculation."""
        report = DataQualityReport(total_records=10, valid_records=8)
        assert report.quality_score == 80.0
        
        report = DataQualityReport(total_records=0, valid_records=0)
        assert report.quality_score == 0.0

    def test_report_error_types(self):
        """Test error type extraction from report."""
        results = [
            ValidationResult(is_valid=False, error_type="ERROR_1"),
            ValidationResult(is_valid=False, error_type="ERROR_2"),
            ValidationResult(is_valid=False, error_type="ERROR_1"),  # Duplicate
            ValidationResult(is_valid=True),
        ]
        
        report = DataQualityReport(validation_results=results)
        error_types = report.error_types
        assert error_types == {"ERROR_1", "ERROR_2"}


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_data_quality_convenience(self):
        """Test validate_data_quality convenience function."""
        base_time = datetime.utcnow()
        
        transactions = [
            {
                "id": "tx_1",
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "source_account": "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
                "fee": 100,
                "amount": 100.0,
            }
            for i in range(2)
        ]
        
        report = validate_data_quality(transactions)
        assert isinstance(report, DataQualityReport)
        assert report.total_records == 2

    def test_check_temporal_consistency_convenience(self):
        """Test check_temporal_consistency convenience function."""
        base_time = datetime.utcnow()
        
        transactions = [
            {"id": f"tx_{i}", "timestamp": (base_time + timedelta(hours=i)).isoformat()}
            for i in range(3)
        ]
        
        results = check_temporal_consistency(transactions)
        assert isinstance(results, list)
        assert len(results) == 2  # ordering + future check
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_check_referential_integrity_convenience(self):
        """Test check_referential_integrity convenience function."""
        transactions = [
            {
                "id": "tx_1",
                "source_account": "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
                "asset_code": "XLM",
                "ledger_sequence": 123,
            }
        ]
        
        results = check_referential_integrity(transactions)
        assert isinstance(results, list)
        assert len(results) == 3  # account + asset + ledger checks
        assert all(isinstance(r, ValidationResult) for r in results)


# Test fixtures for pytest

@pytest.fixture
def sample_transactions() -> List[Dict[str, Any]]:
    """Sample transactions for testing."""
    base_time = datetime.utcnow()
    return [
        {
            "id": f"tx_{i}",
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "source_account": "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "asset_code": "XLM",
            "ledger_sequence": 100 + i,
            "fee": 100,
            "amount": 100.0 * (i + 1),
            "operation_count": 1,
        }
        for i in range(5)
    ]


@pytest.fixture
def invalid_transactions() -> List[Dict[str, Any]]:
    """Invalid transactions for testing."""
    return [
        {
            "id": "tx_invalid_1",
            "source_account": "INVALID_ACCOUNT",
            "asset_code": "invalid_asset",
            "ledger_sequence": -1,
            "fee": -100,
            "amount": -50.0,
            "operation_count": 0,
        },
        {
            "id": "tx_invalid_2",
            "timestamp": "invalid-timestamp",
            "source_account": "gabcd1234567890abcdefghijklmnopqrstuvwxyz1234567890",  # lowercase
            "asset_code": "TOOLONGASSETCODE123",
            "ledger_sequence": "not_a_number",
            "fee": "free",
            "amount": float('nan'),
            "operation_count": 200,
        }
    ]


class TestIntegrationWithFixtures:
    """Integration tests using fixtures."""

    def test_sample_transactions_pass_validation(self, sample_transactions):
        """Test that sample transactions pass validation."""
        validator = DataQualityValidator()
        report = validator.validate_batch(sample_transactions)
        
        # Most validations should pass for sample data
        temporal_results = [r for r in report.validation_results if r.is_valid]
        assert len(temporal_results) > 0

    def test_invalid_transactions_fail_validation(self, invalid_transactions):
        """Test that invalid transactions fail validation."""
        validator = DataQualityValidator()
        report = validator.validate_batch(invalid_transactions)
        
        # Should detect multiple errors
        error_results = [r for r in report.validation_results if not r.is_valid]
        assert len(error_results) > 0
        
        # Check for specific error types
        error_types = {r.error_type for r in error_results if r.error_type}
        assert any(error_type in error_types for error_type in [
            "INVALID_ACCOUNT_FORMAT", "INVALID_ASSET_FORMAT", "NEGATIVE_FEE",
            "NEGATIVE_AMOUNT", "INVALID_OPERATION_COUNT"
        ])
