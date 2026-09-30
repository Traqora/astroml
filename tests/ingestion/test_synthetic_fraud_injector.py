import json
from pathlib import Path
from astroml.ingestion.synthetic_fraud_injector import (
    inject_synthetic_fraud,
    SybilConfig,
    WashLoopConfig,
)

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

    augmented, summary = inject_synthetic_fraud(
        transactions,
        seed=42,
        sybil=sybil_cfg,
        wash=wash_cfg,
        source_field="source",
        dest_field="destination",
        amount_field="amount",
        timestamp_field="timestamp"
    )

    # Save to a golden file locally to pin it
    golden_path = Path(__file__).parent / "golden_fraud_output.json"
    
    # If the file doesn't exist, this is a first run (developer generates it)
    # We'll just write it directly if it doesn't exist, but since it's a test we should provide it.
    output_data = {
        "summary": {
            "original_transactions": summary.original_transactions,
            "sybil_transactions": summary.sybil_transactions,
            "wash_loop_transactions": summary.wash_loop_transactions,
            "injected_transactions": summary.injected_transactions,
            "total_transactions": summary.total_transactions,
        },
        "augmented": augmented
    }

    if not golden_path.exists():
        golden_path.write_text(json.dumps(output_data, indent=2))

    # Read the golden file
    golden_data = json.loads(golden_path.read_text())

    # Assert match
    assert output_data["summary"] == golden_data["summary"], "Injection summary regression"
    assert len(output_data["augmented"]) == len(golden_data["augmented"]), "Transaction count regression"
    
    for i, (actual, expected) in enumerate(zip(output_data["augmented"], golden_data["augmented"])):
        assert actual == expected, f"Mismatch at transaction {i}: {actual} != {expected}"

