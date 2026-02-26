"""Parse Horizon API JSON responses into SQLAlchemy ORM models.

Each ``parse_*`` function accepts a dict (decoded JSON from a Horizon SSE
event) and returns the corresponding ORM model instance.  These functions
perform no I/O and are safe to call from any context.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from astroml.db.schema import Ledger, Operation, Transaction


def _parse_datetime(iso_string: str) -> datetime:
    """Parse an ISO 8601 timestamp from Horizon into a timezone-aware datetime."""
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))


def parse_ledger(data: dict) -> Ledger:
    """Parse a Horizon ledger JSON dict into a Ledger ORM instance."""
    return Ledger(
        sequence=int(data["sequence"]),
        hash=data["hash"],
        prev_hash=data.get("prev_hash"),
        closed_at=_parse_datetime(data["closed_at"]),
        successful_transaction_count=int(data.get("successful_transaction_count", 0)),
        failed_transaction_count=int(data.get("failed_transaction_count", 0)),
        operation_count=int(data.get("operation_count", 0)),
        total_coins=float(data["total_coins"]) if data.get("total_coins") else None,
        fee_pool=float(data["fee_pool"]) if data.get("fee_pool") else None,
        base_fee_in_stroops=int(data["base_fee_in_stroops"]) if data.get("base_fee_in_stroops") else None,
        protocol_version=int(data["protocol_version"]) if data.get("protocol_version") else None,
    )


def parse_transaction(data: dict) -> Transaction:
    """Parse a Horizon transaction JSON dict into a Transaction ORM instance."""
    return Transaction(
        hash=data["hash"],
        ledger_sequence=int(data["ledger"]),
        source_account=data["source_account"],
        created_at=_parse_datetime(data["created_at"]),
        fee=int(data["fee_charged"]),
        operation_count=int(data["operation_count"]),
        successful=bool(data["successful"]),
        memo_type=data.get("memo_type"),
        memo=data.get("memo"),
    )


def parse_operation(data: dict, application_order: int = 1) -> Operation:
    """Parse a Horizon operation JSON dict into an Operation ORM instance.

    Args:
        data: Decoded JSON from Horizon operation response.
        application_order: Position of this operation within its transaction.
    """
    op_type = data["type"]
    destination = _extract_destination(data, op_type)
    amount = _extract_amount(data)
    asset_code, asset_issuer = _extract_asset(data)

    common_keys = {
        "id", "paging_token", "transaction_successful", "source_account",
        "type", "type_i", "created_at", "transaction_hash", "_links",
    }
    details = {k: v for k, v in data.items() if k not in common_keys}

    return Operation(
        id=int(data["id"]),
        transaction_hash=data["transaction_hash"],
        application_order=application_order,
        type=op_type,
        source_account=data["source_account"],
        destination_account=destination,
        amount=float(amount) if amount is not None else None,
        asset_code=asset_code,
        asset_issuer=asset_issuer,
        created_at=_parse_datetime(data["created_at"]),
        details=details if details else None,
    )


def _extract_destination(data: dict, op_type: str) -> Optional[str]:
    """Extract destination account from various operation types."""
    if "to" in data:
        return data["to"]
    if op_type == "create_account" and "account" in data:
        return data["account"]
    if op_type == "account_merge" and "into" in data:
        return data["into"]
    return data.get("destination_account")


def _extract_amount(data: dict) -> Optional[str]:
    """Extract amount from various operation types."""
    if "amount" in data:
        return data["amount"]
    if "starting_balance" in data:
        return data["starting_balance"]
    return None


def _extract_asset(data: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract asset code and issuer from various operation types."""
    asset_type = data.get("asset_type")
    if asset_type == "native":
        return ("XLM", None)
    return (data.get("asset_code"), data.get("asset_issuer"))
