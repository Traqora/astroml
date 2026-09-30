import json

from astroml.ingestion.synthetic_fraud_injector import (
    SybilConfig,
    WashLoopConfig,
    inject_synthetic_fraud,
    run_injection,
)


def test_inject_synthetic_fraud_counts_and_markers():
    base = [
        {
            "source_account": "A",
            "destination_account": "B",
            "amount": 10,
            "created_at": "2025-01-01T00:00:00Z",
        }
    ]

    out, summary = inject_synthetic_fraud(
        base,
        seed=123,
        sybil=SybilConfig(clusters=1, cluster_size=2, tx_per_member=2, base_amount=15.0),
        wash=WashLoopConfig(loops=1, loop_size=3, rounds=2, base_amount=50.0),
    )

    assert summary.original_transactions == 1
    assert summary.sybil_transactions == 4  # 1 * 2 * 2
    assert summary.wash_loop_transactions == 6  # 1 * 3 * 2
    assert summary.injected_transactions == 10
    assert summary.total_transactions == 11

    injected = [tx for tx in out if tx.get("synthetic_fraud")]
    assert len(injected) == 10
    assert {tx["fraud_pattern"] for tx in injected} == {"sybil_cluster", "wash_trading_loop"}


def test_run_injection_keeps_jsonl_format_and_writes_summary(tmp_path):
    input_path = tmp_path / "ledger.jsonl"
    output_path = tmp_path / "ledger_augmented.jsonl"
    summary_path = tmp_path / "summary.json"

    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_account": "S1",
                        "destination_account": "D1",
                        "amount": 1.2,
                        "created_at": "2025-01-02T03:04:05Z",
                    }
                ),
                json.dumps(
                    {
                        "source_account": "S2",
                        "destination_account": "D2",
                        "amount": 2.3,
                        "created_at": "2025-01-02T03:04:06Z",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_injection(
        input_path=str(input_path),
        output_path=str(output_path),
        summary_path=str(summary_path),
        seed=7,
        sybil=SybilConfig(clusters=1, cluster_size=1, tx_per_member=1, base_amount=9.0),
        wash=WashLoopConfig(loops=1, loop_size=2, rounds=1, base_amount=10.0),
        source_field="source_account",
        dest_field="destination_account",
        amount_field="amount",
        timestamp_field="created_at",
    )

    out_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(out_lines) == summary.total_transactions
    assert len(out_lines) == 5  # 2 originals + (1 sybil + 2 wash)

    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["injected_transactions"] == 3
