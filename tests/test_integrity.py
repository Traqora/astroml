"""Unit tests for integrity validation utilities."""
import pytest

from astroml.validation import integrity


class TestIntegrityValidator:
    """Tests for IntegrityValidator class."""

    def test_valid_transactions(self):
        """Should accept valid, unique transactions."""
        v = integrity.IntegrityValidator(required_fields={"id"})
        txs = [
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
            {"id": "2", "payload": "test2", "timestamp": "2024-01-02"},
        ]
        result = v.process(txs)
        assert result.is_valid is True
        assert len(result.valid) == 2
        assert len(result.corrupted) == 0
        assert len(result.duplicates) == 0

    def test_detect_duplicates(self):
        """Should detect duplicate transactions."""
        v = integrity.IntegrityValidator(required_fields={"id"})
        txs = [
            {"id": "1", "payload": "test", "timestamp": "2024-01-01"},
            {"id": "1", "payload": "test", "timestamp": "2024-01-01"},
        ]
        result = v.process(txs)
        assert result.is_valid is True
        assert len(result.valid) == 1
        assert len(result.duplicates) == 1

    def test_detect_corruption(self):
        """Should detect corrupted transactions."""
        v = integrity.IntegrityValidator(required_fields={"id", "timestamp"})
        txs = [
            {"id": "1", "timestamp": "2024-01-01"},
            {"id": "2"},  # missing timestamp
        ]
        result = v.process(txs)
        assert result.is_valid is False
        assert len(result.corrupted) == 1

    def test_mixed_batch(self):
        """Should handle mixed valid, duplicate, and corrupted transactions."""
        v = integrity.IntegrityValidator(required_fields={"id"})
        txs = [
            {"id": "1", "payload": "test1"},
            {"id": "2", "payload": "test2"},
            {"id": "1", "payload": "test1"},  # duplicate
            {"id": "3"},  # missing payload - if required
        ]
        # Only require id, so all should pass validation
        result = v.process(txs)
        assert len(result.valid) == 2
        assert len(result.duplicates) == 1

    def test_strict_mode(self):
        """Should raise error in strict mode on corruption."""
        v = integrity.IntegrityValidator(required_fields={"id"}, strict=True)
        txs = [{"id": "1"}, {}]  # second has no id
        with pytest.raises(integrity.IntegrityError):
            v.process(txs)

    def test_verify_integrity(self):
        """Should verify integrity of transactions."""
        v = integrity.IntegrityValidator(required_fields={"id"})
        valid_txs = [
            {"id": "1", "payload": "test1"},
            {"id": "2", "payload": "test2"},
        ]
        assert v.verify_integrity(valid_txs) is True

        # Add duplicate
        txs_with_dup = [
            {"id": "1", "payload": "test1"},
            {"id": "1", "payload": "test1"},
        ]
        assert v.verify_integrity(txs_with_dup) is False


class TestCheckIntegrity:
    """Tests for check_integrity convenience function."""

    def test_check_integrity_function(self):
        """Should check integrity of transactions."""
        txs = [
            {"id": "1", "payload": "test1"},
            {"id": "2", "payload": "test2"},
        ]
        result = integrity.check_integrity(txs)
        assert result.is_valid is True


class TestFilterValidTransactions:
    """Tests for filter_valid_transactions convenience function."""

    def test_filter_valid(self):
        """Should filter to return only valid transactions."""
        txs = [
            {"id": "1", "payload": "test1"},
            {"id": "2", "payload": "test2"},
            {"id": "1", "payload": "test1"},  # duplicate
        ]
        valid = integrity.filter_valid_transactions(txs)
        assert len(valid) == 2
