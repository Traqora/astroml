"""
Integration tests — accounts (issue #264).

Tests cover: fixture availability, sample data shape, and future CRUD stubs.
These are designed to wire into CI immediately and expand as the accounts API
endpoint is implemented.
"""
import pytest


@pytest.mark.xdist_group("api_accounts")
class TestAccountFixtures:
    """Verify shared fixtures are correctly wired."""

    def test_sample_accounts_count(self, sample_accounts):
        assert len(sample_accounts) == 3

    def test_sample_accounts_have_required_fields(self, sample_accounts):
        for acc in sample_accounts:
            assert "account_id" in acc
            assert "sequence" in acc

    def test_account_ids_are_unique(self, sample_accounts):
        ids = [a["account_id"] for a in sample_accounts]
        assert len(ids) == len(set(ids)), "account IDs must be unique in test fixtures"

    def test_db_session_is_isolated(self, db_session):
        """Each test gets a fresh session — no cross-test state."""
        assert db_session is not None
        # Session should be clean (nothing committed yet)
        assert db_session.new == set()


@pytest.mark.xdist_group("api_accounts")
class TestAccountPagination:
    """Stubs for pagination tests (expand when endpoint exists)."""

    def test_pagination_fixture_supports_slicing(self, sample_accounts):
        page_size = 2
        page_1 = sample_accounts[:page_size]
        page_2 = sample_accounts[page_size:]
        assert len(page_1) == 2
        assert len(page_2) == 1
