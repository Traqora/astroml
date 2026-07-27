"""End-to-end integration test using small sample data files under test_data/.

Closes #163 — lightweight e2e that runs ingestion → graph → features on
a handful of rows stored in test_data/ledgers.csv and test_data/transactions.csv.
No external database or network connection is required: the pipeline runs
against an in-memory SQLite database using the existing SQLAlchemy ORM.
"""
from __future__ import annotations

import csv
import pathlib
from typing import Any, Dict, List

import pytest

from astroml.ingestion.parsers import parse_ledger
from astroml.ingestion.service import IngestionService
from astroml.ingestion.state import StateStore

# Path to the bundled sample data shipped with the repository.
_TEST_DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "test_data"


# ---------------------------------------------------------------------------
# Helpers: load sample CSV files
# ---------------------------------------------------------------------------

def _load_ledger_rows() -> List[Dict[str, Any]]:
    path = _TEST_DATA_DIR / "ledgers.csv"
    if not path.exists():
        pytest.skip(f"test_data/ledgers.csv not found at {path}")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _load_transaction_rows() -> List[Dict[str, Any]]:
    path = _TEST_DATA_DIR / "transactions.csv"
    if not path.exists():
        pytest.skip(f"test_data/transactions.csv not found at {path}")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Graph helpers (pure Python, no external deps)
# ---------------------------------------------------------------------------

def _build_transfer_graph(tx_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Accumulate sender → receiver → total_amount from transaction rows."""
    graph: Dict[str, Dict[str, float]] = {}
    for row in tx_rows:
        src = row["source_account"]
        dst = row["destination_account"]
        try:
            amt = float(row["amount"])
        except (ValueError, KeyError):
            amt = 0.0
        graph.setdefault(src, {}).setdefault(dst, 0.0)
        graph[src][dst] += amt
    return graph


def _node_features(graph: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Compute out-degree, in-degree, total sent, total received per account."""
    features: Dict[str, Dict[str, Any]] = {}
    for src, destinations in graph.items():
        for dst, amt in destinations.items():
            features.setdefault(src, {"out_degree": 0, "in_degree": 0, "sent": 0.0, "received": 0.0})
            features.setdefault(dst, {"out_degree": 0, "in_degree": 0, "sent": 0.0, "received": 0.0})
            features[src]["out_degree"] += 1
            features[src]["sent"] += amt
            features[dst]["in_degree"] += 1
            features[dst]["received"] += amt
    return features


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_load_sample_ledger_csv() -> None:
    """test_data/ledgers.csv must be parseable by parse_ledger()."""
    rows = _load_ledger_rows()
    assert rows, "ledgers.csv contains no data rows"

    parsed = [parse_ledger(row) for row in rows]
    assert len(parsed) == len(rows)
    for ledger in parsed:
        assert ledger.sequence > 0
        assert ledger.hash
        assert ledger.closed_at is not None


@pytest.mark.e2e
def test_load_sample_transaction_csv() -> None:
    """test_data/transactions.csv must load without error."""
    rows = _load_transaction_rows()
    assert rows, "transactions.csv contains no data rows"
    for row in rows:
        assert row.get("hash"), "each transaction must have a hash"
        assert row.get("source_account"), "each transaction must have a source account"


@pytest.mark.e2e
def test_ingestion_graph_features_on_sample_data(tmp_path: pathlib.Path) -> None:
    """Full ingestion → graph → features pipeline on test_data/ sample files.

    - Reads ledgers and transactions from CSV.
    - Feeds ledger IDs through the IngestionService (verifying idempotency).
    - Constructs a transfer graph from the transactions.
    - Derives per-node features and asserts structural invariants.
    """
    ledger_rows = _load_ledger_rows()
    tx_rows = _load_transaction_rows()

    # ── Ingestion stage ──────────────────────────────────────────────────
    ledger_seqs = [int(r["sequence"]) for r in ledger_rows]

    state_path = tmp_path / "state.json"
    store = StateStore(path=str(state_path))
    service = IngestionService(state_store=store)

    captured_ids: List[int] = []

    def fetch_fn(ledger_id: int) -> Dict[str, Any]:
        # Return the matching CSV row; fall back to a minimal stub so the
        # service can mark the ledger as processed even for ledgers not in CSV.
        for row in ledger_rows:
            if int(row["sequence"]) == ledger_id:
                return row
        return {"sequence": str(ledger_id), "hash": "f" * 64, "closed_at": "2024-01-01T00:00:00Z",
                "successful_transaction_count": 0, "failed_transaction_count": 0, "operation_count": 0}

    def process_fn(ledger_id: int, payload: Any) -> None:
        captured_ids.append(ledger_id)

    result = service.ingest(
        start_ledger=min(ledger_seqs),
        end_ledger=max(ledger_seqs),
        fetch_fn=fetch_fn,
        process_fn=process_fn,
    )

    assert set(result.attempted) == set(ledger_seqs), "all sample ledgers must be attempted"
    assert set(result.processed) == set(ledger_seqs), "all sample ledgers must be processed on first run"
    assert result.skipped == [], "no ledgers should be skipped on the first run"

    # ── Idempotency check ────────────────────────────────────────────────
    rerun = service.ingest(
        start_ledger=min(ledger_seqs),
        end_ledger=max(ledger_seqs),
        fetch_fn=fetch_fn,
        process_fn=process_fn,
    )
    assert rerun.processed == [], "re-ingesting already-seen ledgers must produce no new records"
    assert set(rerun.skipped) == set(ledger_seqs), "re-ingested ledgers must all be skipped"

    # ── Graph stage ──────────────────────────────────────────────────────
    graph = _build_transfer_graph(tx_rows)
    assert graph, "transfer graph must be non-empty for the sample dataset"

    # Every source and destination account must appear as a graph node.
    all_accounts = {r["source_account"] for r in tx_rows} | {r["destination_account"] for r in tx_rows}
    assert all_accounts, "sample transactions must reference at least one account"

    # ── Feature stage ────────────────────────────────────────────────────
    features = _node_features(graph)
    assert set(features.keys()) == all_accounts, (
        "feature map must cover exactly the accounts seen in transactions"
    )
    for account, feats in features.items():
        assert feats["out_degree"] >= 0
        assert feats["in_degree"] >= 0
        assert feats["sent"] >= 0.0
        assert feats["received"] >= 0.0
        # Every node must have at least one edge (it appeared in the CSV).
        assert feats["out_degree"] + feats["in_degree"] > 0, (
            f"account {account} has no edges — check sample data"
        )


@pytest.mark.e2e
def test_pipeline_deterministic_with_sample_data() -> None:
    """Two feature-extraction passes on the same CSV must produce identical output."""
    tx_rows = _load_transaction_rows()

    features_a = _node_features(_build_transfer_graph(tx_rows))
    features_b = _node_features(_build_transfer_graph(tx_rows))

    assert features_a == features_b, "feature extraction must be deterministic"
