"""Transaction normalizer for extracting structured data from Horizon operations.

Two entry points cover the full operation surface:

- :func:`normalize_operation` — the general-purpose path. Use this for every
  operation type *except* path payments (``path_payment_strict_send`` /
  ``path_payment_strict_receive``). It always returns exactly one
  :class:`~astroml.db.schema.NormalizedTransaction`.
- :func:`normalize_path_payment_hops` — the path-payment-aware path. Path
  payments route funds through one or more intermediate assets, so a single
  Horizon operation can represent several distinct graph edges (one per hop).
  This function decomposes the operation into one
  :class:`~astroml.db.schema.NormalizedTransaction` per hop via
  :func:`astroml.ingestion.parsers.extract_path_payment_hops`, and
  transparently falls back to :func:`normalize_operation` for non-path-payment
  types (or if hop extraction finds nothing to expand), so callers that don't
  know the operation type ahead of time can call it uniformly instead of
  branching on ``data["type"]`` themselves.

Also usable as a CLI: ``python -m astroml.ingestion.normalizer --help``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from astroml.db.schema import NormalizedTransaction
from astroml.ingestion.parsers import (
    _PATH_PAYMENT_TYPES,
    _extract_amount,
    _extract_destination,
    _parse_datetime,
    extract_asset_string,
    extract_path_payment_hops,
    ledger_sequence_from_operation_id,
)


def _natural_key(data: dict) -> tuple[int | None, int | None]:
    """Return ``(ledger_sequence, operation_id)`` for a raw operation payload.

    Horizon's operation id is a toid, so the ledger is recovered from it rather
    than read from a separate field.  Payloads with no id yield ``(None, None)``:
    the row is then written without a natural key instead of being keyed on a
    guess, which keeps the ingest path working for synthetic or partial input.
    """
    raw_id = data.get("id")
    if raw_id is None:
        return (None, None)
    operation_id = int(raw_id)
    return (ledger_sequence_from_operation_id(operation_id), operation_id)


def normalize_operation(data: dict) -> NormalizedTransaction:
    """Transform raw horizon operation data into a NormalizedTransaction.

    For path payments use :func:`normalize_path_payment_hops` instead to
    get one record per hop.
    """
    op_type = data["type"]
    sender = data["source_account"]
    receiver = _extract_destination(data, op_type)

    amount_str = _extract_amount(data)
    amount = float(amount_str) if amount_str is not None else None

    normalized_asset = extract_asset_string(data)

    timestamp = _parse_datetime(data["created_at"])
    transaction_hash = data["transaction_hash"]
    ledger_sequence, operation_id = _natural_key(data)

    return NormalizedTransaction(
        transaction_hash=transaction_hash,
        ledger_sequence=ledger_sequence,
        operation_id=operation_id,
        hop_index=0,
        sender=sender,
        receiver=receiver,
        asset=normalized_asset,
        amount=amount,
        timestamp=timestamp,
    )


def normalize_path_payment_hops(data: dict) -> list[NormalizedTransaction]:
    """Return one NormalizedTransaction per hop for a path payment operation.

    Every hop keeps the operation's real ``transaction_hash`` and the hop's
    position in ``hop_index``, which together with the operation id makes each
    hop a separate row that is still idempotent on retry.

    Falls back to a single record (via :func:`normalize_operation`) for
    non-path-payment types so callers can use this function uniformly.
    """
    if data.get("type") not in _PATH_PAYMENT_TYPES:
        return [normalize_operation(data)]

    hops = extract_path_payment_hops(data)
    if not hops:
        return [normalize_operation(data)]

    timestamp = _parse_datetime(data["created_at"])
    transaction_hash = data["transaction_hash"]
    ledger_sequence, operation_id = _natural_key(data)

    return [
        NormalizedTransaction(
            transaction_hash=transaction_hash,
            ledger_sequence=ledger_sequence,
            operation_id=operation_id,
            hop_index=hop["hop_index"],
            sender=hop["from_account"],
            receiver=hop["to_account"],
            asset=hop["asset"],
            amount=hop["amount"],
            timestamp=timestamp,
        )
        for hop in hops
    ]


_SNAPSHOT_FIELDS = ("transaction_hash", "sender", "receiver", "asset", "amount", "timestamp")


def snapshot_transaction(tx: NormalizedTransaction) -> dict[str, Any]:
    """Serialise a NormalizedTransaction into a JSON-safe snapshot dict (issue #978).

    Args:
        tx: The normalized transaction to snapshot.

    Returns:
        Dict with the normalized fields; ``timestamp`` is ISO-8601 and
        ``amount`` is a float (or None).
    """
    return {
        "transaction_hash": tx.transaction_hash,
        "sender": tx.sender,
        "receiver": tx.receiver,
        "asset": tx.asset,
        "amount": float(tx.amount) if tx.amount is not None else None,
        "timestamp": tx.timestamp.isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI — issue #990
# ---------------------------------------------------------------------------

_CLI_DESCRIPTION = (
    "Normalize raw Horizon operation JSON into NormalizedTransaction records. "
    "Input is a JSON object or array of operations; output is one JSON record per line."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m astroml.ingestion.normalizer`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m astroml.ingestion.normalizer",
        description=_CLI_DESCRIPTION,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="path to a JSON file of operations, or '-' for stdin (default: -)",
    )
    parser.add_argument(
        "--hops",
        action="store_true",
        help="expand path payments into one record per hop",
    )
    return parser


def _to_record(tx: NormalizedTransaction) -> dict[str, Any]:
    return {
        "transaction_hash": tx.transaction_hash,
        "sender": tx.sender,
        "receiver": tx.receiver,
        "asset": tx.asset,
        "amount": float(tx.amount) if tx.amount is not None else None,
        "timestamp": tx.timestamp.isoformat(),
    }


def restore_transaction(snapshot: dict[str, Any]) -> NormalizedTransaction:
    """Rebuild a NormalizedTransaction from :func:`snapshot_transaction` output (issue #978).

    Args:
        snapshot: Snapshot dict containing every normalized field.

    Returns:
        A new, unpersisted NormalizedTransaction.

    Raises:
        ValueError: if a required field is missing or the timestamp is invalid.
    """
    missing = [f for f in _SNAPSHOT_FIELDS if f not in snapshot]
    if missing:
        raise ValueError(f"snapshot missing fields: {missing}")
    amount = snapshot["amount"]
    return NormalizedTransaction(
        transaction_hash=snapshot["transaction_hash"],
        sender=snapshot["sender"],
        receiver=snapshot["receiver"],
        asset=snapshot["asset"],
        amount=float(amount) if amount is not None else None,
        timestamp=datetime.fromisoformat(snapshot["timestamp"]),
    )


_RECORD_FIELDS = ("transaction_hash", "sender", "receiver", "asset", "amount", "timestamp")


def restore_record(record: dict[str, Any]) -> NormalizedTransaction:
    """Restore a NormalizedTransaction from a CLI output record (issue #985).

    Inverse of the JSON records printed by :func:`main`, so a normalized
    snapshot written to disk can be reloaded without re-fetching Horizon.

    Args:
        record: One decoded JSON record as emitted by the CLI.

    Returns:
        A new, unpersisted NormalizedTransaction.

    Raises:
        ValueError: if a field is missing or the timestamp is not ISO-8601.
    """
    missing = [f for f in _RECORD_FIELDS if f not in record]
    if missing:
        raise ValueError(f"record missing fields: {missing}")
    amount = record["amount"]
    return NormalizedTransaction(
        transaction_hash=record["transaction_hash"],
        sender=record["sender"],
        receiver=record["receiver"],
        asset=record["asset"],
        amount=float(amount) if amount is not None else None,
        timestamp=datetime.fromisoformat(record["timestamp"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.input == "-":
        raw = sys.stdin.read()
    else:
        with open(args.input) as fh:
            raw = fh.read()
    data = json.loads(raw)
    ops = data if isinstance(data, list) else [data]
    normalize = normalize_path_payment_hops if args.hops else (lambda op: [normalize_operation(op)])
    for op in ops:
        for tx in normalize(op):
            print(json.dumps(_to_record(tx)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
