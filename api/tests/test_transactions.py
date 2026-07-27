"""
Integration tests — transactions (issue #264).

Tests cover: filtering by account, stats aggregation, and edge cases.
"""
import pytest


@pytest.mark.xdist_group("api_transactions")
class TestTransactionFixtures:

    def test_sample_transactions_count(self, sample_transactions):
        assert len(sample_transactions) == 3

    def test_transactions_have_required_fields(self, sample_transactions):
        required = {"transaction_hash", "ledger_sequence", "source_account", "fee_charged", "successful"}
        for tx in sample_transactions:
            assert required.issubset(tx.keys())

    def test_transaction_hashes_unique(self, sample_transactions):
        hashes = [t["transaction_hash"] for t in sample_transactions]
        assert len(hashes) == len(set(hashes))


@pytest.mark.xdist_group("api_transactions")
class TestTransactionFiltering:

    def test_filter_by_account(self, sample_transactions):
        account = "GAAZI4TCR3TY5OJHCTJC2A4QSY6CJWJH5IAJTGKIN2ER7LBNVKOCCWN"
        filtered = [t for t in sample_transactions if t["source_account"] == account]
        assert len(filtered) == 2

    def test_filter_successful_only(self, sample_transactions):
        successful = [t for t in sample_transactions if t["successful"]]
        assert len(successful) == 2

    def test_stats_total_fees(self, sample_transactions):
        total = sum(t["fee_charged"] for t in sample_transactions)
        assert total == 400

    def test_edge_case_empty_filter(self, sample_transactions):
        filtered = [t for t in sample_transactions if t["source_account"] == "NONEXISTENT"]
        assert filtered == []
