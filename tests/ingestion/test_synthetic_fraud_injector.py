"""Golden-file regression tests for the synthetic fraud injector (issue #874).

The original test covered exactly one combined scenario (sybil clusters +
wash-trading loops together, seed=42) against ``golden_fraud_output.json``.
That golden entry is left byte-for-byte unchanged here — overwriting an
established golden file on a "convenient" run would silently mask a real
regression, which is exactly what this test exists to catch.

Added below: two more deterministic golden scenarios exercising the
injector's other supported configurations in isolation (sybil-only, with
zero wash loops; wash-only, with zero sybil clusters) — i.e. the *individual*
fraud pattern types the combined scenario never isolates from each other —
plus a structural (non-golden) test for the empty-input-ledger edge case,
which is excluded from the golden-file pattern because
``inject_synthetic_fraud`` falls back to ``datetime.now()`` for its
timestamp base when there is no input transaction to derive one from,
making that one scenario's output inherently non-deterministic across runs.
"""

import json
from pathlib import Path

from astroml.ingestion.synthetic_fraud_injector import (
    SybilConfig,
    WashLoopConfig,
    inject_synthetic_fraud,
)

GOLDEN_DIR = Path(__file__).parent


def _run_injection(transactions, seed, sybil, wash):
    augmented, summary = inject_synthetic_fraud(
        transactions,
        seed=seed,
        sybil=sybil,
        wash=wash,
        source_field="source",
        dest_field="destination",
        amount_field="amount",
        timestamp_field="timestamp",
    )
    return {
        "summary": {
            "original_transactions": summary.original_transactions,
            "sybil_transactions": summary.sybil_transactions,
            "wash_loop_transactions": summary.wash_loop_transactions,
            "injected_transactions": summary.injected_transactions,
            "total_transactions": summary.total_transactions,
        },
        "augmented": augmented,
    }


def _assert_matches_golden_file(output_data: dict, golden_path: Path) -> None:
    # If the file doesn't exist, this is a first run (developer generates it)
    # We'll just write it directly if it doesn't exist, but since it's a test we should provide it.
    if not golden_path.exists():
        golden_path.write_text(json.dumps(output_data, indent=2))

    golden_data = json.loads(golden_path.read_text())

    assert output_data["summary"] == golden_data["summary"], "Injection summary regression"
    assert len(output_data["augmented"]) == len(
        golden_data["augmented"]
    ), "Transaction count regression"

    for i, (actual, expected) in enumerate(zip(output_data["augmented"], golden_data["augmented"])):
        assert actual == expected, f"Mismatch at transaction {i}: {actual} != {expected}"


def test_synthetic_fraud_golden_file(tmp_path):
    # Dummy input
    transactions = [
        {
            "source": "A",
            "destination": "B",
            "amount": 10.0,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    ]

    sybil_cfg = SybilConfig(clusters=2, cluster_size=3, tx_per_member=2, base_amount=10.0)
    wash_cfg = WashLoopConfig(loops=2, loop_size=3, rounds=2, base_amount=50.0)

    output_data = _run_injection(transactions, seed=42, sybil=sybil_cfg, wash=wash_cfg)
    _assert_matches_golden_file(output_data, GOLDEN_DIR / "golden_fraud_output.json")


def test_synthetic_fraud_golden_file_sybil_only():
    """Sybil clusters in isolation (zero wash loops) — the combined scenario
    above never exercises ``wash.loops=0``, so this is genuinely new
    coverage of the injector's sybil-only code path."""
    transactions = [
        {
            "source": "A",
            "destination": "B",
            "amount": 10.0,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    ]

    sybil_cfg = SybilConfig(clusters=1, cluster_size=2, tx_per_member=1, base_amount=5.0)
    wash_cfg = WashLoopConfig(loops=0, loop_size=3, rounds=2, base_amount=50.0)

    output_data = _run_injection(transactions, seed=11, sybil=sybil_cfg, wash=wash_cfg)

    assert output_data["summary"]["sybil_transactions"] > 0
    assert output_data["summary"]["wash_loop_transactions"] == 0
    assert all(
        tx.get("fraud_pattern") != "wash_trading_loop"
        for tx in output_data["augmented"]
        if "fraud_pattern" in tx
    )

    _assert_matches_golden_file(output_data, GOLDEN_DIR / "golden_fraud_output_sybil_only.json")


def test_synthetic_fraud_golden_file_wash_only():
    """Wash-trading loops in isolation (zero sybil clusters) — the combined
    scenario above never exercises ``sybil.clusters=0``."""
    transactions = [
        {
            "source": "A",
            "destination": "B",
            "amount": 10.0,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    ]

    sybil_cfg = SybilConfig(clusters=0, cluster_size=3, tx_per_member=2, base_amount=25.0)
    wash_cfg = WashLoopConfig(loops=1, loop_size=3, rounds=2, base_amount=75.0)

    output_data = _run_injection(transactions, seed=23, sybil=sybil_cfg, wash=wash_cfg)

    assert output_data["summary"]["wash_loop_transactions"] > 0
    assert output_data["summary"]["sybil_transactions"] == 0
    assert all(
        tx.get("fraud_pattern") != "sybil_cluster"
        for tx in output_data["augmented"]
        if "fraud_pattern" in tx
    )

    _assert_matches_golden_file(output_data, GOLDEN_DIR / "golden_fraud_output_wash_only.json")


def test_synthetic_fraud_empty_input_ledger():
    """Edge case: injecting fraud patterns into an empty ledger.

    Deliberately NOT a golden-file test: with zero input transactions,
    ``_max_timestamp`` has nothing to derive a base timestamp from and falls
    back to ``datetime.now(timezone.utc)``, so the injected transactions'
    timestamps are wall-clock-dependent and would make an exact-match golden
    file flaky. Instead this asserts the structural/count invariants that
    *are* deterministic regardless of wall-clock time.
    """
    sybil_cfg = SybilConfig(clusters=1, cluster_size=2, tx_per_member=1, base_amount=10.0)
    wash_cfg = WashLoopConfig(loops=1, loop_size=3, rounds=1, base_amount=20.0)

    output_data = _run_injection([], seed=5, sybil=sybil_cfg, wash=wash_cfg)

    assert output_data["summary"]["original_transactions"] == 0
    assert (
        output_data["summary"]["total_transactions"]
        == output_data["summary"]["injected_transactions"]
    )
    assert output_data["summary"]["injected_transactions"] == (
        output_data["summary"]["sybil_transactions"]
        + output_data["summary"]["wash_loop_transactions"]
    )
    assert len(output_data["augmented"]) == output_data["summary"]["total_transactions"]
    # No original/clean transactions should appear since none were provided.
    assert all(tx.get("synthetic_fraud") is True for tx in output_data["augmented"])
