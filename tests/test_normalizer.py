"""Tests for astroml.ingestion.normalizer."""

from datetime import datetime, timezone

import pytest

from astroml.db.schema import NormalizedTransaction
from astroml.ingestion.normalizer import normalize_operation, normalize_path_payment_hops


@pytest.fixture()
def sample_payment_json():
    return {
        "id": "123",
        "type": "payment",
        "source_account": "G_SENDER",
        "to": "G_RECEIVER",
        "amount": "100.5",
        "asset_type": "native",
        "created_at": "2024-01-15T12:30:00Z",
        "transaction_hash": "a_hash",
    }


@pytest.fixture()
def sample_create_account_json():
    return {
        "id": "456",
        "type": "create_account",
        "source_account": "G_FUNDER",
        "account": "G_NEW_ACCOUNT",
        "starting_balance": "1000.0",
        "asset_type": "native",
        "created_at": "2024-01-15T12:35:00Z",
        "transaction_hash": "b_hash",
    }


@pytest.fixture()
def sample_trustline_json():
    return {
        "id": "789",
        "type": "change_trust",
        "source_account": "G_TRUSTOR",
        "asset_type": "credit_alphanum4",
        "asset_code": "USDC",
        "asset_issuer": "G_ISSUER",
        "created_at": "2024-01-15T12:40:00Z",
        "transaction_hash": "c_hash",
    }


def test_normalize_payment(sample_payment_json):
    norm = normalize_operation(sample_payment_json)

    assert isinstance(norm, NormalizedTransaction)
    assert norm.sender == "G_SENDER"
    assert norm.receiver == "G_RECEIVER"
    assert norm.amount == 100.5
    assert norm.asset == "XLM"
    assert norm.transaction_hash == "a_hash"
    assert norm.timestamp == datetime(2024, 1, 15, 12, 30, tzinfo=timezone.utc)


def test_normalize_create_account(sample_create_account_json):
    norm = normalize_operation(sample_create_account_json)

    assert norm.sender == "G_FUNDER"
    assert norm.receiver == "G_NEW_ACCOUNT"
    assert norm.amount == 1000.0
    assert norm.asset == "XLM"
    assert norm.transaction_hash == "b_hash"


def test_normalize_other_operation(sample_trustline_json):
    norm = normalize_operation(sample_trustline_json)

    assert norm.sender == "G_TRUSTOR"
    assert norm.receiver is None
    assert norm.amount is None
    assert norm.asset == "USDC:G_ISSUER"
    assert norm.transaction_hash == "c_hash"


# ---------------------------------------------------------------------------
# normalize_path_payment_hops (#946)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_path_payment_json():
    return {
        "id": "999",
        "type": "path_payment_strict_send",
        "source_account": "G_SENDER",
        "to": "G_RECEIVER",
        "source_amount": "50.0",
        "destination_amount": "48.5",
        "amount": "48.5",
        "asset_type": "credit_alphanum4",
        "asset_code": "USDC",
        "asset_issuer": "G_ISSUER",
        "source_asset_type": "native",
        "path": [],
        "created_at": "2024-01-15T13:00:00Z",
        "transaction_hash": "path_hash",
    }


@pytest.fixture()
def sample_multi_hop_path_payment_json(sample_path_payment_json):
    return {
        **sample_path_payment_json,
        "path": [{"asset_type": "credit_alphanum4", "asset_code": "BTC", "asset_issuer": "G_MID"}],
    }


def test_normalize_path_payment_hops_single_hop(sample_path_payment_json):
    """A path payment with no intermediate assets produces exactly one hop."""
    hops = normalize_path_payment_hops(sample_path_payment_json)

    assert len(hops) == 1
    hop = hops[0]
    assert isinstance(hop, NormalizedTransaction)
    assert hop.sender == "G_SENDER"
    assert hop.receiver == "G_RECEIVER"
    # With no intermediate path assets, the single hop is both the first and
    # last hop; extract_path_payment_hops reports the *source* leg's asset
    # (the sending asset), not the destination asset the receiver ends up with.
    assert hop.asset == "XLM"
    # Same reasoning as above: the first-hop branch is checked before the
    # last-hop branch, so a single-hop payment reports the source amount.
    assert hop.amount == 50.0
    assert hop.transaction_hash == "path_hash_hop0"
    assert hop.timestamp == datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc)


def test_normalize_path_payment_hops_multi_hop(sample_multi_hop_path_payment_json):
    """A path payment with one intermediate asset produces two ordered hops."""
    hops = normalize_path_payment_hops(sample_multi_hop_path_payment_json)

    assert len(hops) == 2

    first, second = hops
    assert first.sender == "G_SENDER"
    assert first.receiver == "__path__path_hash_1"
    assert first.asset == "XLM"
    assert first.amount == 50.0
    assert first.transaction_hash == "path_hash_hop0"

    assert second.sender == "__path__path_hash_1"
    assert second.receiver == "G_RECEIVER"
    # The middle of the chain is the intermediate path asset (BTC), not the
    # destination asset (USDC) — the destination asset only applies to the
    # final hop when there's more than one intermediate asset in the path.
    assert second.asset == "BTC:G_MID"
    assert second.amount == 48.5
    assert second.transaction_hash == "path_hash_hop1"

    # Both hops share the same transaction timestamp.
    assert first.timestamp == second.timestamp == datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc)


def _fields(tx: NormalizedTransaction) -> tuple:
    return (tx.transaction_hash, tx.sender, tx.receiver, tx.asset, tx.amount, tx.timestamp)


def test_normalize_path_payment_hops_falls_back_for_non_path_payment(sample_payment_json):
    """Non-path-payment types fall back to normalize_operation, returning one record."""
    hops = normalize_path_payment_hops(sample_payment_json)

    assert len(hops) == 1
    assert _fields(hops[0]) == _fields(normalize_operation(sample_payment_json))


def test_normalize_path_payment_hops_falls_back_when_no_hops_extracted(monkeypatch):
    """If hop extraction returns nothing for a path-payment type, fall back to
    normalize_operation rather than silently dropping the transaction."""
    import astroml.ingestion.normalizer as normalizer_module

    monkeypatch.setattr(normalizer_module, "extract_path_payment_hops", lambda data: [])

    data = {
        "id": "1",
        "type": "path_payment_strict_receive",
        "source_account": "G_SENDER",
        "to": "G_RECEIVER",
        "destination_amount": "10.0",
        "asset_type": "native",
        "created_at": "2024-01-15T14:00:00Z",
        "transaction_hash": "fallback_hash",
    }

    hops = normalize_path_payment_hops(data)

    assert len(hops) == 1
    assert _fields(hops[0]) == _fields(normalize_operation(data))
