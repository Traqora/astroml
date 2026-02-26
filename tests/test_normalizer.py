"""Tests for astroml.ingestion.normalizer."""
from datetime import datetime, timezone

import pytest

from astroml.db.schema import NormalizedTransaction
from astroml.ingestion.normalizer import normalize_operation


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
