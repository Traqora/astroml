"""Snapshot/restore regression tests for the normalizer (issue #978)."""

import json
from datetime import datetime, timezone

import pytest

from astroml.ingestion.normalizer import restore_transaction, snapshot_transaction
from astroml.db.schema import NormalizedTransaction

TS = datetime(2024, 5, 1, 12, 30, tzinfo=timezone.utc)


def _tx(**overrides):
    fields = dict(
        transaction_hash="abc123",
        sender="GSENDER",
        receiver="GRECEIVER",
        asset="XLM",
        amount=12.5,
        timestamp=TS,
    )
    fields.update(overrides)
    return NormalizedTransaction(**fields)


def _fields(tx):
    return (tx.transaction_hash, tx.sender, tx.receiver, tx.asset, tx.amount, tx.timestamp)


def test_round_trip_preserves_fields():
    tx = _tx()
    assert _fields(restore_transaction(snapshot_transaction(tx))) == _fields(tx)


def test_round_trip_through_json():
    tx = _tx()
    restored = restore_transaction(json.loads(json.dumps(snapshot_transaction(tx))))
    assert _fields(restored) == _fields(tx)
    assert restored.timestamp.tzinfo is not None


def test_round_trip_with_null_receiver_and_amount():
    tx = _tx(receiver=None, amount=None)
    assert _fields(restore_transaction(snapshot_transaction(tx))) == _fields(tx)


def test_round_trip_preserves_natural_key_and_hop_index():
    tx = _tx(ledger_sequence=123, operation_id=456, hop_index=2)
    restored = restore_transaction(json.loads(json.dumps(snapshot_transaction(tx))))

    assert (restored.ledger_sequence, restored.operation_id, restored.hop_index) == (123, 456, 2)


def test_restore_accepts_legacy_snapshot_without_natural_key():
    snapshot = snapshot_transaction(_tx())
    for field in ("ledger_sequence", "operation_id", "hop_index"):
        snapshot.pop(field)

    restored = restore_transaction(snapshot)

    assert (restored.ledger_sequence, restored.operation_id, restored.hop_index) == (None, None, 0)


def test_restore_rejects_missing_fields():
    snap = snapshot_transaction(_tx())
    del snap["timestamp"]
    with pytest.raises(ValueError, match="timestamp"):
        restore_transaction(snap)


def test_restore_rejects_bad_timestamp():
    snap = snapshot_transaction(_tx())
    snap["timestamp"] = "not-a-date"
    with pytest.raises(ValueError):
        restore_transaction(snap)
