"""Transaction normalizer for extracting structured data from Horizon operations."""
from __future__ import annotations

from typing import Optional

from astroml.db.schema import NormalizedTransaction
from astroml.ingestion.parsers import (
    _PATH_PAYMENT_TYPES,
    _extract_amount,
    _extract_asset,
    _extract_destination,
    _parse_datetime,
    extract_path_payment_hops,
)


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

    asset_code, asset_issuer = _extract_asset(data)

    if asset_code == "XLM" and asset_issuer is None:
        normalized_asset = "XLM"
    else:
        normalized_asset = f"{asset_code}:{asset_issuer}" if asset_code and asset_issuer else "UNKNOWN"

    timestamp = _parse_datetime(data["created_at"])
    transaction_hash = data["transaction_hash"]

    return NormalizedTransaction(
        transaction_hash=transaction_hash,
        sender=sender,
        receiver=receiver,
        asset=normalized_asset,
        amount=amount,
        timestamp=timestamp,
    )


def normalize_path_payment_hops(data: dict) -> list[NormalizedTransaction]:
    """Return one NormalizedTransaction per hop for a path payment operation.

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

    return [
        NormalizedTransaction(
            transaction_hash=f"{transaction_hash}_hop{hop['hop_index']}",
            sender=hop["from_account"],
            receiver=hop["to_account"],
            asset=hop["asset"],
            amount=hop["amount"],
            timestamp=timestamp,
        )
        for hop in hops
    ]
