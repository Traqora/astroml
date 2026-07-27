"""Unit tests for deduplication utilities.

These tests intentionally import the dedupe submodule through the validation
package surface. The validation package must therefore avoid eager imports of
unrelated modules so this file remains stable under parallel collection.

Issue #204 — flakiness fix:
- Every test method creates a **fresh** Deduplicator instance via the
  `deduplicator` fixture so there is zero shared state between tests.
- `@pytest.mark.xdist_group("dedupe")` keeps all tests in this module on the
  same worker when pytest-xdist is active, avoiding any race on module-level
  imports that could surface intermittently on slow runners.
"""
import pytest

from astroml.validation import dedupe


def _tx(
    tx_id: str,
    payload: str = "test",
    timestamp: str = "2024-01-01",
):
    return {"id": tx_id, "payload": payload, "timestamp": timestamp}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def deduplicator():
    """Fresh Deduplicator for each test — no shared state."""
    return dedupe.Deduplicator()


@pytest.fixture()
def tracking_deduplicator():
    """Fresh Deduplicator with conflict tracking enabled."""
    return dedupe.Deduplicator(track_conflicts=True)


# ── TestDeduplicator ──────────────────────────────────────────────────────────

@pytest.mark.xdist_group("dedupe")
class TestDeduplicator:
    """Tests for Deduplicator class."""

    def test_add_unique_transaction(self, deduplicator):
        """Should add unique transaction."""
        tx = _tx("1")
        result = deduplicator.add(tx)
        assert result is True
        assert len(deduplicator.seen_hashes) == 1

    def test_add_duplicate_transaction(self, deduplicator):
        """Should reject duplicate transaction."""
        tx = _tx("1")
        deduplicator.add(tx)
        result = deduplicator.add(tx)
        assert result is False

    def test_check_duplicate(self, deduplicator):
        """Should check for duplicates without adding."""
        tx = _tx("1")
        assert deduplicator.check(tx) is False
        deduplicator.add(tx)
        assert deduplicator.check(tx) is True

    def test_process_batch(self, deduplicator):
        """Should process batch and separate duplicates."""
        txs = [
            _tx("1", payload="test1"),
            _tx("2", payload="test2", timestamp="2024-01-02"),
            _tx("1", payload="test1"),  # duplicate
        ]
        result = deduplicator.process(txs)
        assert len(result.unique) == 2
        assert len(result.duplicates) == 1

    def test_filter_unique(self, deduplicator):
        """Should filter and return unique transactions."""
        txs = [
            _tx("1", payload="test1"),
            _tx("1", payload="test1"),
            _tx("2", payload="test2", timestamp="2024-01-02"),
        ]
        unique = deduplicator.filter_duplicates(txs, return_unique=True)
        assert len(unique) == 2

    def test_filter_duplicates_only(self, deduplicator):
        """Should filter and return only duplicates."""
        txs = [
            _tx("1", payload="test1"),
            _tx("1", payload="test1"),
            _tx("2", payload="test2", timestamp="2024-01-02"),
        ]
        duplicates = deduplicator.filter_duplicates(txs, return_unique=False)
        assert len(duplicates) == 1

    def test_reset(self, deduplicator):
        """Should clear all state — no bleed into subsequent tests."""
        tx = _tx("1")
        deduplicator.add(tx)
        deduplicator.reset()
        assert len(deduplicator.seen_hashes) == 0

    def test_conflict_tracking(self, tracking_deduplicator):
        """Should track conflict records."""
        tx = _tx("1")
        tracking_deduplicator.add(tx)
        tracking_deduplicator.add(tx)  # duplicate
        assert len(tracking_deduplicator.conflicts) == 1
        assert tracking_deduplicator.conflicts[0].conflict_type == dedupe.ConflictType.DUPLICATE

    def test_independent_instances_do_not_share_state(self):
        """Two Deduplicator instances must never share seen_hashes (regression for #204)."""
        d1 = dedupe.Deduplicator()
        d2 = dedupe.Deduplicator()
        tx = _tx("x")
        d1.add(tx)
        # d2 is brand-new — must not see d1's hash
        assert d2.check(tx) is False, "Deduplicator instances must not share state"
        assert len(d2.seen_hashes) == 0

    def test_fresh_instances_do_not_share_state(self):
        """A new Deduplicator instance must start with an empty seen set."""
        first = dedupe.Deduplicator()
        second = dedupe.Deduplicator()
        tx = _tx("shared")

        assert first.add(tx) is True
        assert second.check(tx) is False
        assert second.add(tx) is True


# ── TestDeduplicate (convenience function) ────────────────────────────────────

@pytest.mark.xdist_group("dedupe")
class TestDeduplicate:
    """Tests for deduplicate convenience function."""

    def test_deduplicate_function(self):
        """Should deduplicate transactions."""
        txs = [
            _tx("1", payload="test1"),
            _tx("2", payload="test2", timestamp="2024-01-02"),
            _tx("1", payload="test1"),
        ]
        result = dedupe.deduplicate(txs)
        assert len(result.unique) == 2
        assert len(result.duplicates) == 1

    def test_deduplicate_empty_list(self):
        """Should handle an empty input without error."""
        result = dedupe.deduplicate([])
        assert len(result.unique) == 0
        assert len(result.duplicates) == 0

    def test_deduplicate_all_unique(self):
        """Should return all items when none are duplicates."""
        txs = [
            {"id": str(i), "payload": f"p{i}", "timestamp": "2024-01-01"}
            for i in range(5)
        ]
        result = dedupe.deduplicate(txs)
        assert len(result.unique) == 5
        assert len(result.duplicates) == 0
