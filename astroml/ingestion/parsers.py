"""Parse Horizon API JSON responses into SQLAlchemy ORM models.

Each ``parse_*`` function accepts a dict (decoded JSON from a Horizon SSE
event) and returns the corresponding ORM model instance.  These functions
perform no I/O and are safe to call from any context.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from astroml.db.schema import Effect, Ledger, Operation, Transaction


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


def parse_effect(data: dict) -> Effect:
    """Parse a Horizon effect JSON dict into an Effect ORM instance."""
    effect_type = data.get("type", "")
    
    # Extract common fields
    account = data.get("account")
    
    # Extract type-specific fields
    amount = None
    asset_code = None
    asset_issuer = None
    destination = None
    
    if effect_type in ["account_created", "account_credited", "account_debited"]:
        amount = data.get("amount")
        if amount:
            asset_type = data.get("asset_type")
            if asset_type == "native":
                asset_code = "XLM"
                asset_issuer = None
            else:
                asset_code = data.get("asset_code")
                asset_issuer = data.get("asset_issuer")
    
    if effect_type == "account_credited":
        destination = account
    
    # Store all non-common fields in details
    common_keys = {
        "id", "paging_token", "account", "type", "created_at", "_links"
    }
    details = {k: v for k, v in data.items() if k not in common_keys}
    
    return Effect(
        id=int(data["id"]),
        account=account,
        type=effect_type,
        amount=float(amount) if amount is not None else None,
        asset_code=asset_code,
        asset_issuer=asset_issuer,
        destination_account=destination,
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
    # For path payments: prefer destination_amount (what receiver gets)
    if "destination_amount" in data:
        return data["destination_amount"]
    if "source_amount" in data:
        return data["source_amount"]
    return None


def _extract_asset(data: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract asset code and issuer from various operation types."""
    asset_type = data.get("asset_type")
    if asset_type == "native":
        return ("XLM", None)
    return (data.get("asset_code"), data.get("asset_issuer"))


def extract_path_payment_hops(data: dict) -> list[dict]:
    """Decompose a path payment into ordered per-hop dicts.

    Each hop dict has keys: from_account, to_account, asset_code,
    asset_issuer, amount, hop_index, is_first_hop, is_last_hop.

    Returns an empty list for non-path-payment operations.
    """
    if data.get("type") not in _PATH_PAYMENT_TYPES:
        return []

    sender = data["source_account"]
    receiver = _extract_destination(data, data["type"])
    path = data.get("path", [])  # intermediate assets

    # Build asset chain: [source_asset, ...path_assets..., dest_asset]
    def _asset_str(asset_dict: dict) -> str:
        if asset_dict.get("asset_type") == "native":
            return "XLM"
        code = asset_dict.get("asset_code", "UNKNOWN")
        issuer = asset_dict.get("asset_issuer", "")
        return f"{code}:{issuer}" if issuer else code

    src_asset_type = data.get("source_asset_type", data.get("asset_type", ""))
    if src_asset_type == "native":
        src_asset = "XLM"
    else:
        src_code = data.get("source_asset_code", data.get("asset_code", "UNKNOWN"))
        src_issuer = data.get("source_asset_issuer", data.get("asset_issuer", ""))
        src_asset = f"{src_code}:{src_issuer}" if src_issuer else src_code

    dst_asset_type = data.get("asset_type", "")
    if dst_asset_type == "native":
        dst_asset = "XLM"
    else:
        dst_code = data.get("asset_code", "UNKNOWN")
        dst_issuer = data.get("asset_issuer", "")
        dst_asset = f"{dst_code}:{dst_issuer}" if dst_issuer else dst_code

    path_assets = [_asset_str(p) for p in path]
    asset_chain = [src_asset] + path_assets + [dst_asset]

    # Amounts: source_amount on first hop, destination_amount on last hop,
    # None for intermediate hops (not exposed by Horizon).
    src_amount = data.get("source_amount")
    dst_amount = data.get("destination_amount", data.get("amount"))

    # Intermediate accounts are not exposed by Horizon; use sentinel "__path__"
    # so the graph builder can distinguish them from real accounts.
    n_hops = len(asset_chain) - 1
    hops = []
    for i in range(n_hops):
        from_acc = sender if i == 0 else f"__path__{data['transaction_hash']}_{i}"
        to_acc = receiver if i == n_hops - 1 else f"__path__{data['transaction_hash']}_{i + 1}"
        amount = src_amount if i == 0 else (dst_amount if i == n_hops - 1 else None)
        hops.append({
            "from_account": from_acc,
            "to_account": to_acc,
            "asset": asset_chain[i],
            "amount": float(amount) if amount is not None else None,
            "hop_index": i,
            "is_first_hop": i == 0,
            "is_last_hop": i == n_hops - 1,
        })
    return hops
