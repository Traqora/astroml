"""Transaction normalizer for extracting structured data from Horizon operations."""
from __future__ import annotations

from typing import Optional

from astroml.db.schema import NormalizedTransaction
from astroml.ingestion.parsers import (
    _extract_amount,
    _extract_asset,
    _extract_destination,
    _parse_datetime,
)


def normalize_operation(data: dict) -> NormalizedTransaction:
    """Transform raw horizon operation data into a NormalizedTransaction."""
    op_type = data["type"]
    sender = data["source_account"]
    receiver = _extract_destination(data, op_type)
    
    amount_str = _extract_amount(data)
    amount = float(amount_str) if amount_str is not None else None
    
    asset_code, asset_issuer = _extract_asset(data)
    
    if asset_code == "XLM" and asset_issuer is None:
        normalized_asset = "XLM"
    else:
        # Default fallback to "UNKNOWN" if no explicit asset info is found
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
