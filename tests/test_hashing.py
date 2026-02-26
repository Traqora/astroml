"""Unit tests for hashing utilities."""
import pytest

from astroml.validation import hashing


class TestComputeTransactionHash:
    """Tests for compute_transaction_hash function."""

    def test_deterministic_hash(self):
        """Hash should be deterministic for same input."""
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        hash1 = hashing.compute_transaction_hash(tx)
        hash2 = hashing.compute_transaction_hash(tx)
        assert hash1 == hash2

    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different hashes."""
        tx1 = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        tx2 = {"id": "2", "payload": "test", "timestamp": "2024-01-01"}
        hash1 = hashing.compute_transaction_hash(tx1)
        hash2 = hashing.compute_transaction_hash(tx2)
        assert hash1 != hash2

    def test_sorted_keys_deterministic(self):
        """Hash should be deterministic regardless of key order."""
        tx1 = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        tx2 = {"timestamp": "2024-01-01", "id": "1", "payload": "test"}
        hash1 = hashing.compute_transaction_hash(tx1)
        hash2 = hashing.compute_transaction_hash(tx2)
        assert hash1 == hash2

    def test_custom_fields(self):
        """Should only hash specified fields."""
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01", "extra": "data"}
        hash_all = hashing.compute_transaction_hash(tx)
        hash_partial = hashing.compute_transaction_hash(tx, fields={"id"})
        assert hash_all != hash_partial
        # Hash with only id should match
        assert hash_partial == hashing.compute_transaction_hash({"id": "1"})

    def test_stored_hash_mismatch(self, caplog):
        """Should log warning on hash mismatch."""
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        wrong_hash = "wrong_hash_value"
        result = hashing.compute_transaction_hash(tx, stored_hash=wrong_hash)
        assert result is not None
        # Should log warning about mismatch
        assert "mismatch" in caplog.text.lower()


class TestVerifyTransactionHash:
    """Tests for verify_transaction_hash function."""

    def test_valid_hash(self):
        """Should return True for matching hash."""
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        expected_hash = hashing.compute_transaction_hash(tx)
        assert hashing.verify_transaction_hash(tx, expected_hash) is True

    def test_invalid_hash(self):
        """Should return False for non-matching hash."""
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        wrong_hash = "wrong_hash_value"
        assert hashing.verify_transaction_hash(tx, wrong_hash) is False


class TestHashBatch:
    """Tests for hash_batch function."""

    def test_batch_hashes(self):
        """Should compute hashes for all transactions."""
        txs = [
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
            {"id": "2", "payload": "test2", "timestamp": "2024-01-02"},
        ]
        hashes = hashing.hash_batch(txs)
        assert len(hashes) == 2
        assert hashes[0] != hashes[1]

    def test_empty_batch(self):
        """Should handle empty batch."""
        hashes = hashing.hash_batch([])
        assert hashes == []
