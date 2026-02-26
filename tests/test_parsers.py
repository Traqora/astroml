"""Tests for astroml.ingestion.parsers."""
from datetime import datetime, timezone

import pytest

from astroml.ingestion.parsers import (
    _parse_datetime,
    parse_ledger,
    parse_operation,
    parse_transaction,
)


# -- Fixtures: sample Horizon JSON responses ----------------------------------


@pytest.fixture()
def sample_ledger_json():
    return {
        "sequence": 12345,
        "hash": "a" * 64,
        "prev_hash": "b" * 64,
        "closed_at": "2024-01-15T12:30:00Z",
        "successful_transaction_count": 5,
        "failed_transaction_count": 1,
        "operation_count": 10,
        "total_coins": "100000000000.0000000",
        "fee_pool": "1234.5670000",
        "base_fee_in_stroops": 100,
        "protocol_version": 20,
        "paging_token": "12345",
    }


@pytest.fixture()
def sample_transaction_json():
    return {
        "hash": "c" * 64,
        "ledger": 12345,
        "source_account": "G" + "A" * 55,
        "created_at": "2024-01-15T12:30:00Z",
        "fee_charged": "100",
        "operation_count": 2,
        "successful": True,
        "memo_type": "none",
        "paging_token": "53919970611200",
    }


@pytest.fixture()
def sample_payment_json():
    return {
        "id": "53919970611201",
        "paging_token": "53919970611201",
        "transaction_successful": True,
        "source_account": "G" + "A" * 55,
        "type": "payment",
        "type_i": 1,
        "created_at": "2024-01-15T12:30:00Z",
        "transaction_hash": "c" * 64,
        "to": "G" + "B" * 55,
        "from": "G" + "A" * 55,
        "amount": "100.0000000",
        "asset_type": "native",
        "_links": {},
    }


@pytest.fixture()
def sample_create_account_json():
    return {
        "id": "53919970611202",
        "paging_token": "53919970611202",
        "transaction_successful": True,
        "source_account": "G" + "A" * 55,
        "type": "create_account",
        "type_i": 0,
        "created_at": "2024-01-15T12:30:00Z",
        "transaction_hash": "c" * 64,
        "account": "G" + "C" * 55,
        "funder": "G" + "A" * 55,
        "starting_balance": "1000.0000000",
        "_links": {},
    }


# -- Tests: datetime parsing --------------------------------------------------


def test_parse_datetime_z_suffix():
    dt = _parse_datetime("2024-01-15T12:30:00Z")
    assert dt == datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)


def test_parse_datetime_is_timezone_aware():
    dt = _parse_datetime("2024-01-15T12:30:00Z")
    assert dt.tzinfo is not None


# -- Tests: parse_ledger ------------------------------------------------------


def test_parse_ledger_maps_fields(sample_ledger_json):
    ledger = parse_ledger(sample_ledger_json)
    assert ledger.sequence == 12345
    assert ledger.hash == "a" * 64
    assert ledger.prev_hash == "b" * 64
    assert ledger.successful_transaction_count == 5
    assert ledger.failed_transaction_count == 1
    assert ledger.operation_count == 10
    assert ledger.protocol_version == 20


def test_parse_ledger_closed_at_is_datetime(sample_ledger_json):
    ledger = parse_ledger(sample_ledger_json)
    assert isinstance(ledger.closed_at, datetime)


# -- Tests: parse_transaction -------------------------------------------------


def test_parse_transaction_maps_core_fields(sample_transaction_json):
    tx = parse_transaction(sample_transaction_json)
    assert tx.hash == "c" * 64
    assert tx.ledger_sequence == 12345
    assert tx.fee == 100
    assert tx.operation_count == 2
    assert tx.successful is True


def test_parse_transaction_memo_none(sample_transaction_json):
    tx = parse_transaction(sample_transaction_json)
    assert tx.memo is None


# -- Tests: parse_operation ---------------------------------------------------


def test_parse_payment_destination(sample_payment_json):
    op = parse_operation(sample_payment_json)
    assert op.destination_account == "G" + "B" * 55
    assert op.amount == 100.0
    assert op.asset_code == "XLM"
    assert op.asset_issuer is None


def test_parse_create_account_destination(sample_create_account_json):
    op = parse_operation(sample_create_account_json)
    assert op.destination_account == "G" + "C" * 55
    assert op.amount == 1000.0
    assert op.type == "create_account"


def test_parse_operation_details_captures_extra(sample_payment_json):
    op = parse_operation(sample_payment_json)
    assert op.details is not None
    assert "from" in op.details


def test_parse_operation_id_is_int(sample_payment_json):
    op = parse_operation(sample_payment_json)
    assert isinstance(op.id, int)
    assert op.id == 53919970611201
