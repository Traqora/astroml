"""Parse Horizon API JSON responses into SQLAlchemy ORM models.

See ADR-003 (docs/adr/003-polars-ingestion.md) for Polars ingestion framework choices.

Each ``parse_*`` function accepts a dict (decoded JSON from a Horizon SSE
event) and returns the corresponding ORM model instance.  These functions
perform no I/O and are safe to call from any context.
"""

from __future__ import annotations

from datetime import datetime

from astroml.db.schema import Effect, Ledger, Operation, Transaction

# Path payment operation types from Horizon
_PATH_PAYMENT_TYPES = {
    "path_payment_strict_send",
    "path_payment_strict_receive",
}


def _parse_datetime(iso_string: str) -> datetime:
    """Parse an ISO 8601 timestamp from Horizon into a timezone-aware datetime."""
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))


#: Bit widths of the fields packed into a Stellar toid.  The id of every
#: ledger, transaction, operation and effect is ``ledger << 32 | tx << 12 | op``
#: (see https://developers.stellar.org/docs/learn/glossary#total-order-id),
#: which is what makes an operation id usable as an ingest-time natural key.
_TOD_LEDGER_SHIFT = 32


def ledger_sequence_from_operation_id(operation_id: int) -> int:
    """Recover the ledger sequence an operation was applied in from its toid.

    ``operation_id`` is Horizon's operation id, which packs the ledger, the
    transaction's application order within it, and the operation's index within
    the transaction.  Only the ledger is needed to key an ingestion write.

    >>> ledger_sequence_from_operation_id(53919970611201)
    12554
    """
    return int(operation_id) >> _TOD_LEDGER_SHIFT


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
        base_fee_in_stroops=(
            int(data["base_fee_in_stroops"]) if data.get("base_fee_in_stroops") else None
        ),
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
        "id",
        "paging_token",
        "transaction_successful",
        "source_account",
        "type",
        "type_i",
        "created_at",
        "transaction_hash",
        "_links",
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
    common_keys = {"id", "paging_token", "account", "type", "created_at", "_links"}
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


def _extract_destination(data: dict, op_type: str) -> str | None:
    """Extract destination account from various operation types."""
    if "to" in data:
        return data["to"]
    if op_type == "create_account" and "account" in data:
        return data["account"]
    if op_type == "account_merge" and "into" in data:
        return data["into"]
    return data.get("destination_account")


def _extract_amount(data: dict) -> str | None:
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


def _extract_asset(data: dict) -> tuple[str | None, str | None]:
    """Extract asset code and issuer from various operation types."""
    asset_type = data.get("asset_type")
    if asset_type == "native":
        return ("XLM", None)
    return (data.get("asset_code"), data.get("asset_issuer"))


def extract_asset_string(data: dict, prefix: str = "") -> str:
    """Extract and format an asset string consistently as code:issuer or XLM."""
    asset_type = data.get(f"{prefix}asset_type", data.get("asset_type", ""))
    if asset_type == "native":
        return "XLM"
    
    code = data.get(f"{prefix}asset_code", data.get("asset_code"))
    issuer = data.get(f"{prefix}asset_issuer", data.get("asset_issuer"))
    
    if code == "XLM" and not issuer:
        return "XLM"
    elif code and issuer:
        return f"{code}:{issuer}"
    elif code:
        return str(code)
    else:
        return "UNKNOWN"


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

    src_asset = extract_asset_string(data, prefix="source_")
    dst_asset = extract_asset_string(data)
    path_assets = [extract_asset_string(p) for p in path]
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
        hops.append(
            {
                "from_account": from_acc,
                "to_account": to_acc,
                "asset": asset_chain[i],
                "amount": float(amount) if amount is not None else None,
                "hop_index": i,
                "is_first_hop": i == 0,
                "is_last_hop": i == n_hops - 1,
            }
        )
    return hops
