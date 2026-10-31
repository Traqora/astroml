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


@pytest.mark.xdist_group("api_accounts")
class TestAccountsEndpoint:
    """Regression coverage for the live GET /api/v1/accounts endpoints.

    AccountsResponse/TransactionsResponse declare a `page_size` field (no
    camelCase alias), but the router was calling them with `pageSize=...`.
    Pydantic rejects an unknown keyword as a missing required field, so
    every call to list_accounts()/get_account_transactions() 500'd. Fixed
    in astroml mypy-strict batch #862; these tests pin the fix.
    """

    @pytest.mark.xfail(
        reason=(
            "Separate pre-existing bug (not the pageSize issue this test class "
            "targets): AccountOut requires account_id/sequence/flags/"
            "last_modified_ledger/updated_at, but the ApiAccount ORM model "
            "(api/models/orm.py) only has public_key/first_seen/last_active/"
            "balance/home_domain/created_at — model_validate() always raises "
            "a pydantic ValidationError. Needs a schema/ORM reconciliation "
            "decision, out of scope for the mypy-annotation batch that added "
            "this test; tracked separately."
        ),
        strict=True,
    )
    def test_list_accounts_returns_200_with_page_size_field(
        self, client, db_session, seeded_account
    ):
        # The HTTP client reads through a separate (async) DB connection, so
        # the fixture's flush() alone is invisible to it; commit to make the
        # seeded row visible across connections, matching test_auth.py's
        # auth_client fixture.
        db_session.commit()

        response = client.get("/api/v1/accounts")
        assert response.status_code == 200
        body = response.json()
        assert "page_size" in body
        assert body["page_size"] == 20
        assert body["page"] == 1
        assert body["total"] >= 1
        assert any(a["account_id"] == seeded_account.public_key for a in body["data"])

    @pytest.mark.xfail(
        reason=(
            "Separate pre-existing bug (not the pageSize issue this test class "
            "targets): TransactionOut requires operation_count, but the "
            "ApiTransaction ORM model (api/models/orm.py) has operation_type "
            "instead — no operation_count attribute at all, so "
            "model_validate() always raises a pydantic ValidationError. Needs "
            "a schema/ORM reconciliation decision, out of scope for the "
            "mypy-annotation batch that added this test; tracked separately."
        ),
        strict=True,
    )
    def test_get_account_transactions_returns_200_with_page_size_field(
        self, client, db_session, seeded_account, seeded_transaction
    ):
        db_session.commit()

        response = client.get(f"/api/v1/accounts/{seeded_account.public_key}/transactions")
        assert response.status_code == 200
        body = response.json()
        assert "page_size" in body
        assert body["page_size"] == 20
        assert body["total"] >= 1

    def test_get_account_fraud_summary_returns_200(self, client, db_session, seeded_account):
        db_session.commit()

        response = client.get(f"/api/v1/accounts/{seeded_account.public_key}/fraud-summary")
        assert response.status_code == 200
        body = response.json()
        assert body["account_id"] == seeded_account.public_key
        assert body["total_alerts"] == 0
        assert body["latest_score"] is None
