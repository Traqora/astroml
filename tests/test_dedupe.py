"""Unit tests for deduplication utilities."""
import pytest

from astroml.validation import dedupe


class TestDeduplicator:
    """Tests for Deduplicator class."""

    def test_add_unique_transaction(self):
        """Should add unique transaction."""
        dedup = dedupe.Deduplicator()
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        result = dedup.add(tx)
        assert result is True
        assert "1" in dedup.seen_hashes

    def test_add_duplicate_transaction(self):
        """Should reject duplicate transaction."""
        dedup = dedupe.Deduplicator()
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        dedup.add(tx)
        result = dedup.add(tx)
        assert result is False

    def test_check_duplicate(self):
        """Should check for duplicates without adding."""
        dedup = dedupe.Deduplicator()
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        assert dedup.check(tx) is False
        dedup.add(tx)
        assert dedup.check(tx) is True

    def test_process_batch(self):
        """Should process batch and separate duplicates."""
        dedup = dedupe.Deduplicator()
        txs = [
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
            {"id": "2", "payload": "test2", "timestamp": "2024-01-02"},
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},  # duplicate
        ]
        result = dedup.process(txs)
        assert len(result.unique) == 2
        assert len(result.duplicates) == 1

    def test_filter_unique(self):
        """Should filter and return unique transactions."""
        dedup = dedupe.Deduplicator()
        txs = [
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
            {"id": "2", "payload": "test2", "timestamp": "2024-01-02"},
        ]
        unique = dedup.filter_duplicates(txs, return_unique=True)
        assert len(unique) == 2

    def test_filter_duplicates_only(self):
        """Should filter and return only duplicates."""
        dedup = dedupe.Deduplicator()
        txs = [
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
            {"id": "2", "payload": "test2", "timestamp": "2024-01-02"},
        ]
        duplicates = dedup.filter_duplicates(txs, return_unique=False)
        assert len(duplicates) == 1

    def test_reset(self):
        """Should clear all state."""
        dedup = dedupe.Deduplicator()
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        dedup.add(tx)
        dedup.reset()
        assert len(dedup.seen_hashes) == 0

    def test_conflict_tracking(self):
        """Should track conflict records."""
        dedup = dedupe.Deduplicator(track_conflicts=True)
        tx = {"id": "1", "payload": "test", "timestamp": "2024-01-01"}
        dedup.add(tx)
        dedup.add(tx)  # duplicate
        assert len(dedup.conflicts) == 1
        assert dedup.conflicts[0].conflict_type == dedupe.ConflictType.DUPLICATE


class TestDeduplicate:
    """Tests for deduplicate convenience function."""

    def test_deduplicate_function(self):
        """Should deduplicate transactions."""
        txs = [
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
            {"id": "2", "payload": "test2", "timestamp": "2024-01-02"},
            {"id": "1", "payload": "test1", "timestamp": "2024-01-01"},
        ]
        result = dedupe.deduplicate(txs)
        assert len(result.unique) == 2
        assert len(result.duplicates) == 1
