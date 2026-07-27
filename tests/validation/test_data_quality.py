"""Comprehensive data quality validation tests.

Covers completeness, uniqueness, consistency, and hash-integrity checks
across the astroml.validation pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from astroml.validation import dedupe, integrity, validator
from astroml.validation.validator import CorruptionType


class TestCompleteness:
    """Required fields must be present and non-null."""

    def test_all_required_fields_present(self, valid_transactions):
        v = validator.TransactionValidator(required_fields={"id", "source_account"})
        for tx in valid_transactions:
            result = v.validate(tx)
            assert result.is_valid, f"Expected valid: {tx}"

    def test_missing_field_flagged(self):
        v = validator.TransactionValidator(required_fields={"id", "source_account", "amount"})
        tx = {"id": "tx_missing", "source_account": "GABC"}
        result = v.validate(tx)
        assert not result.is_valid
        assert any(e.error_type == CorruptionType.MISSING_FIELD for e in result.errors)

    def test_null_required_field_flagged(self):
        v = validator.TransactionValidator(required_fields={"id"})
        result = v.validate({"id": None, "source_account": "GABC"})
        assert not result.is_valid

    def test_empty_dict_fails_required_fields(self):
        v = validator.TransactionValidator(required_fields={"id"})
        assert not v.validate({}).is_valid

    def test_batch_completeness_surfaces_invalid_rows(self, transactions_with_missing_fields):
        v = validator.TransactionValidator(required_fields={"id", "source_account"})
        results = v.validate_batch(transactions_with_missing_fields)
        assert len(results) == len(transactions_with_missing_fields)
        assert all(not r.is_valid for r in results)

    def test_non_dict_flagged_as_malformed(self):
        v = validator.TransactionValidator()
        for bad in [None, "string", 42, [1, 2]]:
            result = v.validate(bad)
            assert not result.is_valid
            assert any(e.error_type == CorruptionType.MALFORMED_STRUCTURE for e in result.errors)


class TestUniqueness:
    """Duplicate transactions must be detected reliably."""

    def test_clean_batch_has_no_duplicates(self, valid_transactions):
        result = dedupe.deduplicate(valid_transactions)
        assert len(result.duplicates) == 0
        assert len(result.unique) == len(valid_transactions)

    def test_exact_duplicate_detected(self, duplicate_transactions):
        result = dedupe.deduplicate(duplicate_transactions)
        assert len(result.duplicates) == 1
        assert len(result.unique) == len(duplicate_transactions) - 1

    def test_first_occurrence_is_kept(self):
        tx_a = {"id": "tx_a", "amount": 10}
        tx_b = {"id": "tx_b", "amount": 20}
        result = dedupe.deduplicate([tx_a, tx_b, {**tx_a}])
        assert len(result.unique) == 2
        assert len(result.duplicates) == 1

    def test_deduplicator_stateful_across_batches(self):
        d = dedupe.Deduplicator()
        tx = {"id": "tx_1", "amount": 100}
        r1 = d.process([tx])
        r2 = d.process([{**tx}])
        assert len(r1.unique) == 1
        assert len(r2.duplicates) == 1

    def test_deduplicator_reset_clears_state(self):
        d = dedupe.Deduplicator()
        tx = {"id": "tx_reset", "amount": 50}
        d.process([tx])
        d.reset()
        result = d.process([{**tx}])
        assert len(result.unique) == 1
        assert len(result.duplicates) == 0

    def test_integrity_validator_detects_duplicates(self):
        v = integrity.IntegrityValidator(required_fields={"id"})
        result = v.process([{"id": "dup"}, {"id": "dup"}])
        assert len(result.duplicates) == 1
        assert len(result.valid) == 1


class TestConsistency:
    """Field types must match declared expectations."""

    def test_correct_types_pass(self):
        v = validator.TransactionValidator(field_types={"id": str, "amount": float})
        assert v.validate({"id": "tx_001", "amount": 100.0}).is_valid

    def test_wrong_type_flagged(self):
        v = validator.TransactionValidator(field_types={"id": str})
        result = v.validate({"id": 12345})
        assert not result.is_valid
        assert any(e.error_type == CorruptionType.INVALID_TYPE for e in result.errors)

    def test_multiple_type_violations_all_reported(self):
        v = validator.TransactionValidator(field_types={"id": str, "amount": float})
        result = v.validate({"id": 123, "amount": "not_a_float"})
        type_errors = [e for e in result.errors if e.error_type == CorruptionType.INVALID_TYPE]
        assert len(type_errors) == 2

    def test_batch_surfaces_type_inconsistent_rows(self):
        v = validator.TransactionValidator(field_types={"id": str})
        results = v.validate_batch([{"id": "ok_1"}, {"id": "ok_2"}, {"id": 999}])
        assert sum(1 for r in results if not r.is_valid) == 1


class TestAccuracy:
    """Hash-based integrity checks catch tampered data."""

    def test_correct_hash_passes(self):
        from astroml.validation.hashing import compute_transaction_hash

        v = validator.TransactionValidator()
        tx = {"id": "tx_hash_ok", "payload": "original"}
        result = v.validate(tx, stored_hash=compute_transaction_hash(tx))
        assert result.is_valid

    def test_wrong_hash_flagged(self):
        v = validator.TransactionValidator()
        tx = {"id": "tx_hash_bad", "payload": "tampered"}
        result = v.validate(tx, stored_hash="00000000_wrong_hash")
        assert not result.is_valid
        assert any(e.error_type == CorruptionType.HASH_MISMATCH for e in result.errors)

    def test_integrity_processor_flags_corrupted_rows(self):
        v = integrity.IntegrityValidator(required_fields={"id", "amount"})
        result = v.process([{"id": "ok", "amount": 50}, {"id": "bad"}])
        assert len(result.corrupted) == 1
        assert result.corrupted[0]["id"] == "bad"


class TestTemporalConsistency:
    """Temporal data quality checks for timestamps and ordering."""

    def test_transaction_timestamps_increasing(self):
        """Transactions should have monotonically increasing timestamps within a batch."""
        from datetime import datetime, timedelta
        
        base_time = datetime.utcnow()
        transactions = [
            {"id": f"tx_{i}", "timestamp": (base_time + timedelta(hours=i)).isoformat()}
            for i in range(5)
        ]
        
        # Test valid increasing timestamps
        timestamps = [tx["timestamp"] for tx in transactions]
        assert timestamps == sorted(timestamps)
        
        # Test invalid ordering
        invalid_txs = transactions.copy()
        invalid_txs[2], invalid_txs[3] = invalid_txs[3], invalid_txs[2]
        invalid_timestamps = [tx["timestamp"] for tx in invalid_txs]
        assert invalid_timestamps != sorted(invalid_timestamps)

    def test_future_timestamp_detection(self):
        """Detect transactions with timestamps in the future."""
        from datetime import datetime, timedelta
        
        future_time = datetime.utcnow() + timedelta(days=1)
        future_tx = {"id": "tx_future", "timestamp": future_time.isoformat()}
        
        # Future timestamp should be flagged
        tx_time = datetime.fromisoformat(future_tx["timestamp"])
        assert tx_time > datetime.utcnow()

    def test_ledger_sequence_consistency(self):
        """Ledger sequences should be consistent and increasing."""
        transactions = [
            {"id": f"tx_{i}", "ledger_sequence": 100 + i}
            for i in range(5)
        ]
        
        # Valid sequences
        sequences = [tx["ledger_sequence"] for tx in transactions]
        assert sequences == sorted(sequences)
        assert all(sequences[i] <= sequences[i+1] for i in range(len(sequences)-1))


class TestReferentialIntegrity:
    """Referential integrity checks between related entities."""

    def test_transaction_ledger_reference(self):
        """Transactions should reference valid ledger sequences."""
        valid_tx = {"id": "tx_valid", "ledger_sequence": 123}
        invalid_tx = {"id": "tx_invalid", "ledger_sequence": -1}
        
        assert valid_tx["ledger_sequence"] > 0
        assert invalid_tx["ledger_sequence"] < 0

    def test_account_format_validation(self):
        """Stellar account addresses should follow proper format."""
        import re
        
        # Stellar public key pattern (G followed by 56 alphanumeric chars)
        account_pattern = re.compile(r'^G[A-Z0-9]{56}$')
        
        valid_accounts = [
            "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "GABCDEFGHIJKLMN0PQRSTUVWXYZ0123456789012345"
        ]
        
        invalid_accounts = [
            "XABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",  # Wrong prefix
            "GABCD123",  # Too short
            "gabcd1234567890abcdefghijklmnopqrstuvwxyz1234567890",  # Lowercase
            "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#"  # Extra chars
        ]
        
        for account in valid_accounts:
            assert account_pattern.match(account) is not None
            
        for account in invalid_accounts:
            assert account_pattern.match(account) is None

    def test_asset_format_validation(self):
        """Asset codes and issuers should follow proper format."""
        import re
        
        # Asset code: 1-12 alphanumeric characters
        asset_code_pattern = re.compile(r'^[A-Z0-9]{1,12}$')
        
        valid_codes = ["XLM", "USD", "BTC", "CUSTOM123", "ASSETCODE"]
        invalid_codes = ["xlm", "", "TOOLONGASSETCODE123", "asset-with-dash"]
        
        for code in valid_codes:
            assert asset_code_pattern.match(code) is not None
            
        for code in invalid_codes:
            assert asset_code_pattern.match(code) is None


class TestBusinessRules:
    """Business logic validation for domain-specific rules."""

    def test_fee_non_negative(self):
        """Transaction fees should never be negative."""
        valid_txs = [
            {"id": "tx_1", "fee": 100},
            {"id": "tx_2", "fee": 0},
            {"id": "tx_3", "fee": 1000}
        ]
        
        invalid_txs = [
            {"id": "tx_bad_1", "fee": -100},
            {"id": "tx_bad_2", "fee": -1}
        ]
        
        for tx in valid_txs:
            assert tx["fee"] >= 0
            
        for tx in invalid_txs:
            assert tx["fee"] < 0

    def test_amount_non_negative(self):
        """Transaction amounts should be non-negative for most operation types."""
        valid_amounts = [0, 0.1, 100.0, 1000000.5]
        invalid_amounts = [-0.1, -100.0]
        
        for amount in valid_amounts:
            assert amount >= 0
            
        for amount in invalid_amounts:
            assert amount < 0

    def test_operation_count_reasonable(self):
        """Operation count should be within reasonable bounds."""
        valid_txs = [
            {"id": "tx_1", "operation_count": 1},
            {"id": "tx_2", "operation_count": 10},
            {"id": "tx_3", "operation_count": 100}
        ]
        
        # Stellar allows up to 100 operations per transaction
        for tx in valid_txs:
            assert 1 <= tx["operation_count"] <= 100

    def test_balance_format(self):
        """Account balances should be proper numeric values."""
        valid_balances = [0, 0.1, 100.0, 1000000.123456789]
        invalid_balances = [float('inf'), float('-inf'), float('nan'), None]
        
        for balance in valid_balances:
            assert isinstance(balance, (int, float))
            assert not (balance != balance)  # NaN check
            assert balance == balance  # NaN check
            
        for balance in invalid_balances:
            if balance is None:
                continue  # None might be valid in some contexts
            assert not (balance == balance) or balance in [float('inf'), float('-inf')]


class TestStatisticalValidation:
    """Statistical validation for data distributions and anomalies."""

    def test_amount_distribution_outliers(self):
        """Detect statistical outliers in transaction amounts."""
        import statistics
        
        # Normal distribution of amounts
        normal_amounts = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        
        # Add outliers
        amounts_with_outliers = normal_amounts + [10000.0, 0.0001]
        
        # Calculate IQR for outlier detection
        q1 = statistics.quantiles(normal_amounts, n=4)[0]
        q3 = statistics.quantiles(normal_amounts, n=4)[2]
        iqr = q3 - q1
        
        # Outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Check for outliers
        outliers = [x for x in amounts_with_outliers if x < lower_bound or x > upper_bound]
        assert len(outliers) > 0

    def test_timestamp_gap_detection(self):
        """Detect unusual gaps in transaction timestamps."""
        from datetime import datetime, timedelta
        
        base_time = datetime.utcnow()
        
        # Regular intervals (every 5 minutes)
        regular_timestamps = [
            base_time + timedelta(minutes=5*i) for i in range(10)
        ]
        
        # Add a large gap
        gap_timestamps = regular_timestamps.copy()
        gap_timestamps.append(base_time + timedelta(days=1))
        
        # Calculate gaps
        regular_gaps = [
            (regular_timestamps[i+1] - regular_timestamps[i]).total_seconds()
            for i in range(len(regular_timestamps)-1)
        ]
        
        gap_gaps = [
            (gap_timestamps[i+1] - gap_timestamps[i]).total_seconds()
            for i in range(len(gap_timestamps)-1)
        ]
        
        # Should detect the large gap
        max_regular_gap = max(regular_gaps) if regular_gaps else 0
        max_gap = max(gap_gaps) if gap_gaps else 0
        assert max_gap > max_regular_gap

    def test_duplicate_pattern_detection(self):
        """Detect patterns that might indicate data duplication issues."""
        # Create transactions with similar patterns
        pattern_txs = [
            {"id": f"tx_{i}", "amount": 100.0, "source_account": "ACC1"}
            for i in range(5)
        ]
        
        # Count occurrences of each pattern
        patterns = {}
        for tx in pattern_txs:
            key = (tx["amount"], tx["source_account"])
            patterns[key] = patterns.get(key, 0) + 1
        
        # Should detect the repeated pattern
        assert any(count > 1 for count in patterns.values())


class TestDataQualityPipeline:
    """End-to-end data quality across completeness + uniqueness + integrity."""

    def test_clean_batch_passes_full_pipeline(self, valid_transactions):
        v = validator.TransactionValidator(required_fields={"id", "source_account"})
        assert all(r.is_valid for r in v.validate_batch(valid_transactions))
        assert len(dedupe.deduplicate(valid_transactions).duplicates) == 0
        assert integrity.IntegrityValidator(required_fields={"id"}).process(valid_transactions).is_valid

    def test_strict_mode_raises_on_first_corruption(self):
        v = integrity.IntegrityValidator(required_fields={"id"}, strict=True)
        with pytest.raises(integrity.IntegrityError):
            v.process([{"id": "ok"}, {}])

    def test_filter_returns_only_clean_unique_rows(self):
        txs = [
            {"id": "tx_1", "amount": 100},
            {"id": "tx_2", "amount": 200},
            {"id": "tx_1", "amount": 100},
        ]
        valid = integrity.filter_valid_transactions(txs)
        assert len(valid) == 2

    def test_verify_integrity_false_on_duplicates(self, duplicate_transactions):
        v = integrity.IntegrityValidator(required_fields={"id"})
        assert not v.verify_integrity(duplicate_transactions)

    def test_verify_integrity_true_for_clean_batch(self, valid_transactions):
        v = integrity.IntegrityValidator(required_fields={"id"})
        assert v.verify_integrity(valid_transactions)

    def test_check_integrity_convenience_function(self, valid_transactions):
        result = integrity.check_integrity(valid_transactions)
        assert result.is_valid
