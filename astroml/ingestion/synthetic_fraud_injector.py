from __future__ import annotations

"""Synthetic fraud pattern injector for ledger benchmarking.

This module can copy a "clean" transaction ledger and append synthetic fraud
patterns for controlled benchmarking experiments.

Supported patterns:
- Sybil clusters: one controller account fans out to many coordinated identities.
- Wash-trading loops: a set of accounts repeatedly sends value in a closed loop.

Input and output files support either:
- JSON array of transaction objects
- JSON lines (JSONL), one transaction object per line
"""

import argparse
import json
import pathlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_SOURCE_FIELD = "source_account"
DEFAULT_DEST_FIELD = "destination_account"
DEFAULT_AMOUNT_FIELD = "amount"
DEFAULT_TIMESTAMP_FIELD = "created_at"


@dataclass(frozen=True)
class SybilConfig:
    clusters: int = 2
    cluster_size: int = 5
    tx_per_member: int = 3
    base_amount: float = 25.0


@dataclass(frozen=True)
class WashLoopConfig:
    loops: int = 2
    loop_size: int = 4
    rounds: int = 5
    base_amount: float = 100.0


@dataclass(frozen=True)
class InjectionSummary:
    original_transactions: int
    injected_transactions: int
    total_transactions: int
    sybil_transactions: int
    wash_loop_transactions: int


def _load_transactions(path: pathlib.Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return [], "jsonl"

    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON input must be an array of transaction objects")
        return [dict(item) for item in data], "json"

    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError("Each JSONL line must be a transaction object")
        records.append(parsed)
    return records, "jsonl"


def _write_transactions(path: pathlib.Path, txs: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(txs, indent=2), encoding="utf-8")
        return

    with path.open("w", encoding="utf-8") as f:
        for tx in txs:
            f.write(json.dumps(tx) + "\n")


def _max_timestamp(transactions: list[dict[str, Any]], timestamp_field: str) -> datetime:
    latest: datetime | None = None
    for tx in transactions:
        value = tx.get(timestamp_field)
        if not isinstance(value, str):
            continue
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if latest is None or ts > latest:
            latest = ts

    return latest or datetime.now(timezone.utc)


def _new_tx(
    tx_id: int,
    source: str,
    destination: str,
    amount: float,
    timestamp: datetime,
    source_field: str,
    dest_field: str,
    amount_field: str,
    timestamp_field: str,
    pattern_type: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "synthetic_tx_id": tx_id,
        source_field: source,
        dest_field: destination,
        amount_field: round(amount, 7),
        timestamp_field: timestamp.isoformat().replace("+00:00", "Z"),
        "synthetic_fraud": True,
        "fraud_pattern": pattern_type,
        "fraud_metadata": metadata,
    }


def inject_synthetic_fraud(
    transactions: list[dict[str, Any]],
    *,
    seed: int = 7,
    sybil: SybilConfig = SybilConfig(),
    wash: WashLoopConfig = WashLoopConfig(),
    source_field: str = DEFAULT_SOURCE_FIELD,
    dest_field: str = DEFAULT_DEST_FIELD,
    amount_field: str = DEFAULT_AMOUNT_FIELD,
    timestamp_field: str = DEFAULT_TIMESTAMP_FIELD,
) -> tuple[list[dict[str, Any]], InjectionSummary]:
    """Append synthetic fraud patterns and return augmented ledger + summary."""
    rng = random.Random(seed)
    augmented = [dict(tx) for tx in transactions]

    start_ts = _max_timestamp(augmented, timestamp_field)
    current_ts = start_ts
    next_id = 1

    sybil_injected = 0
    for cluster_idx in range(sybil.clusters):
        controller = f"sybil_controller_{cluster_idx}"
        members = [f"sybil_{cluster_idx}_{i}" for i in range(sybil.cluster_size)]

        for member in members:
            for tx_idx in range(sybil.tx_per_member):
                current_ts += timedelta(seconds=1)
                amount = sybil.base_amount * (1 + rng.uniform(-0.2, 0.2))
                augmented.append(
                    _new_tx(
                        tx_id=next_id,
                        source=controller if tx_idx % 2 == 0 else member,
                        destination=member if tx_idx % 2 == 0 else controller,
                        amount=amount,
                        timestamp=current_ts,
                        source_field=source_field,
                        dest_field=dest_field,
                        amount_field=amount_field,
                        timestamp_field=timestamp_field,
                        pattern_type="sybil_cluster",
                        metadata={
                            "cluster": cluster_idx,
                            "controller": controller,
                            "member": member,
                        },
                    )
                )
                next_id += 1
                sybil_injected += 1

    wash_injected = 0
    for loop_idx in range(wash.loops):
        loop_accounts = [f"wash_{loop_idx}_{i}" for i in range(wash.loop_size)]
        for round_idx in range(wash.rounds):
            for i, sender in enumerate(loop_accounts):
                receiver = loop_accounts[(i + 1) % len(loop_accounts)]
                current_ts += timedelta(seconds=1)
                amount = wash.base_amount * (1 + rng.uniform(-0.03, 0.03))
                augmented.append(
                    _new_tx(
                        tx_id=next_id,
                        source=sender,
                        destination=receiver,
                        amount=amount,
                        timestamp=current_ts,
                        source_field=source_field,
                        dest_field=dest_field,
                        amount_field=amount_field,
                        timestamp_field=timestamp_field,
                        pattern_type="wash_trading_loop",
                        metadata={
                            "loop": loop_idx,
                            "round": round_idx,
                        },
                    )
                )
                next_id += 1
                wash_injected += 1

    summary = InjectionSummary(
        original_transactions=len(transactions),
        injected_transactions=sybil_injected + wash_injected,
        total_transactions=len(augmented),
        sybil_transactions=sybil_injected,
        wash_loop_transactions=wash_injected,
    )
    return augmented, summary


def run_injection(
    *,
    input_path: str,
    output_path: str,
    summary_path: str | None,
    seed: int,
    sybil: SybilConfig,
    wash: WashLoopConfig,
    source_field: str,
    dest_field: str,
    amount_field: str,
    timestamp_field: str,
) -> InjectionSummary:
    """Load input ledger, inject fraud patterns, and save output ledger."""
    in_path = pathlib.Path(input_path)
    out_path = pathlib.Path(output_path)
    txs, fmt = _load_transactions(in_path)

    augmented, summary = inject_synthetic_fraud(
        txs,
        seed=seed,
        sybil=sybil,
        wash=wash,
        source_field=source_field,
        dest_field=dest_field,
        amount_field=amount_field,
        timestamp_field=timestamp_field,
    )

    _write_transactions(out_path, augmented, fmt)

    if summary_path:
        pathlib.Path(summary_path).write_text(
            json.dumps(summary.__dict__, indent=2),
            encoding="utf-8",
        )

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject synthetic fraud patterns into a ledger copy"
    )
    parser.add_argument("--input", required=True, help="Path to clean ledger file (JSON or JSONL)")
    parser.add_argument("--output", required=True, help="Path to augmented output ledger")
    parser.add_argument("--summary", default=None, help="Optional path for JSON summary output")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for deterministic generation")

    parser.add_argument("--sybil-clusters", type=int, default=2)
    parser.add_argument("--sybil-cluster-size", type=int, default=5)
    parser.add_argument("--sybil-tx-per-member", type=int, default=3)
    parser.add_argument("--sybil-base-amount", type=float, default=25.0)

    parser.add_argument("--wash-loops", type=int, default=2)
    parser.add_argument("--wash-loop-size", type=int, default=4)
    parser.add_argument("--wash-rounds", type=int, default=5)
    parser.add_argument("--wash-base-amount", type=float, default=100.0)

    parser.add_argument("--source-field", default=DEFAULT_SOURCE_FIELD)
    parser.add_argument("--destination-field", default=DEFAULT_DEST_FIELD)
    parser.add_argument("--amount-field", default=DEFAULT_AMOUNT_FIELD)
    parser.add_argument("--timestamp-field", default=DEFAULT_TIMESTAMP_FIELD)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: inject fraud patterns into a transactions file.

    Reads ``--input``, writes the augmented dataset to ``--output`` and the
    injection summary to ``--summary``, then prints the summary as JSON.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``. Passable so the
            CLI is testable without touching process state.

    Returns:
        ``0`` on completion. Errors are signalled by the exception below, not
            by a non-zero code — wrap the call site in
            ``raise SystemExit(main())`` and let the traceback surface.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary = run_injection(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        seed=args.seed,
        sybil=SybilConfig(
            clusters=args.sybil_clusters,
            cluster_size=args.sybil_cluster_size,
            tx_per_member=args.sybil_tx_per_member,
            base_amount=args.sybil_base_amount,
        ),
        wash=WashLoopConfig(
            loops=args.wash_loops,
            loop_size=args.wash_loop_size,
            rounds=args.wash_rounds,
            base_amount=args.wash_base_amount,
        ),
        source_field=args.source_field,
        dest_field=args.destination_field,
        amount_field=args.amount_field,
        timestamp_field=args.timestamp_field,
    )
    print(json.dumps(summary.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
