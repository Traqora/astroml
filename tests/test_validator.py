"""Unit tests for transaction validation utilities."""
import pytest

from astroml.validation import validator


class TestTransactionValidator:
    """Tests for TransactionValidator class."""

    def test_valid_transaction(self):
        """Should accept valid transaction."""
        v = validator.TransactionValidator(required_fields={"id"})
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        result = v.validate(tx)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_missing_required_field(self):
        """Should reject transaction with missing required field."""
        v = validator.TransactionValidator(required_fields={"id", "timestamp"})
        tx = {"id": "1"}  # missing timestamp
        result = v.validate(tx)
        assert result.is_valid is False
        assert any(e.error_type == validator.CorruptionType.MISSING_FIELD for e in result.errors)

    def test_null_required_field(self):
        """Should reject transaction with null required field."""
        v = validator.TransactionValidator(required_fields={"id"})
        tx = {"id": None}
        result = v.validate(tx)
        assert result.is_valid is False

    def test_invalid_type(self):
        """Should reject transaction with invalid type."""
        v = validator.TransactionValidator(field_types={"id": str})
        tx = {"id": 123}  # should be str
        result = v.validate(tx)
        assert result.is_valid is False
        assert any(e.error_type == validator.CorruptionType.INVALID_TYPE for e in result.errors)

    def test_malformed_structure(self):
        """Should reject non-dict transaction."""
        v = validator.TransactionValidator()
        result = v.validate("not a dict")
        assert result.is_valid is False
        assert any(e.error_type == validator.CorruptionType.MALFORMED_STRUCTURE for e in result.errors)

    def test_hash_mismatch(self):
        """Should detect hash mismatch."""
        v = validator.TransactionValidator()
        tx = {"id": "1", "payload": "test"}
        result = v.validate(tx, stored_hash="wrong_hash")
        assert result.is_valid is False
        assert any(e.error_type == validator.CorruptionType.HASH_MISMATCH for e in result.errors)

    def test_batch_validation(self):
        """Should validate batch of transactions."""
        v = validator.TransactionValidator(required_fields={"id"})
        txs = [
            {"id": "1"},
            {"id": "2"},
            {},  # missing id
        ]
        results = v.validate_batch(txs)
        assert len(results) == 3
        assert results[0].is_valid is True
        assert results[1].is_valid is True
        assert results[2].is_valid is False


class TestValidateTransaction:
    """Tests for validate_transaction convenience function."""

    def test_validate_valid_transaction(self):
        """Should validate valid transaction."""
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        result = validator.validate_transaction(tx, required_fields={"id"})
        assert result.is_valid is True

    def test_validate_missing_field(self):
        """Should detect missing required field."""
        tx = {"payload": "test"}
        result = validator.validate_transaction(tx, required_fields={"id"})
        assert result.is_valid is False
