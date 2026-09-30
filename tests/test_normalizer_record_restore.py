"""Snapshot/restore regression tests for normalizer CLI records (issue #985)."""

import json

import pytest

from astroml.ingestion import normalizer
from astroml.ingestion.normalizer import normalize_operation, restore_record

OPS = [
    {
        "type": "payment",
        "source_account": "GSENDER",
        "to": "GRECEIVER",
        "amount": "12.5000000",
        "asset_type": "native",
        "created_at": "2024-05-01T12:30:00Z",
        "transaction_hash": "abc123",
    },
    {
        "type": "payment",
        "source_account": "GA",
        "to": "GB",
        "amount": "3",
        "asset_type": "credit_alphanum4",
        "asset_code": "USDC",
        "asset_issuer": "GISSUER",
        "created_at": "2024-05-02T00:00:00Z",
        "transaction_hash": "def456",
    },
]


def _fields(tx):
    return (tx.transaction_hash, tx.sender, tx.receiver, tx.asset, float(tx.amount), tx.timestamp)


def _snapshot_via_cli(tmp_path, capsys):
    src = tmp_path / "ops.json"
    src.write_text(json.dumps(OPS))
    assert normalizer.main([str(src)]) == 0
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_cli_snapshot_restores_to_original(tmp_path, capsys):
    records = _snapshot_via_cli(tmp_path, capsys)
    restored = [restore_record(r) for r in records]
    expected = [normalize_operation(op) for op in OPS]
    assert [_fields(t) for t in restored] == [_fields(t) for t in expected]
    assert all(t.timestamp.tzinfo is not None for t in restored)


def test_restore_keeps_null_receiver_and_amount():
    record = {
        "transaction_hash": "h",
        "sender": "GA",
        "receiver": None,
        "asset": "XLM",
        "amount": None,
        "timestamp": "2024-01-01T00:00:00+00:00",
    }
    tx = restore_record(record)
    assert tx.receiver is None and tx.amount is None


def test_restore_rejects_missing_field(tmp_path, capsys):
    record = _snapshot_via_cli(tmp_path, capsys)[0]
    del record["asset"]
    with pytest.raises(ValueError, match="asset"):
        restore_record(record)


def test_restore_rejects_bad_timestamp(tmp_path, capsys):
    record = _snapshot_via_cli(tmp_path, capsys)[0]
    record["timestamp"] = "yesterday"
    with pytest.raises(ValueError):
        restore_record(record)
