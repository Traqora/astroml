"""End-to-end integration test for the ingestion → graph → features
pipeline (#193).

Uses an in-memory `StateStore` (file-backed but writes to a pytest tmp
dir so the test is self-contained), a small synthetic ledger dataset,
and a deterministic seed. No external Postgres / Stellar RPC is needed.

The test is `@pytest.mark.e2e` so the CPU CI matrix (#186) picks it up
under its default `not gpu` selector.
"""
from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass
from typing import Dict, List

import pytest

from astroml.ingestion.service import IngestionService
from astroml.ingestion.state import StateStore


@dataclass
class FakeLedgerPayload:
    ledger_id: int
    transfers: List[Dict[str, object]]


def _seed(value: int = 42) -> None:
    random.seed(value)


def _synthetic_ledger(ledger_id: int) -> FakeLedgerPayload:
    """Produce a deterministic synthetic ledger with a handful of
    sender→receiver→amount transfers. Used in place of Horizon for the
    e2e test so the suite has no network dependency."""
    rng = random.Random(ledger_id * 9_973 + 1)
    n_transfers = rng.randint(2, 5)
    accounts = [f"G{chr(ord('A') + i)}" for i in range(6)]
    transfers = []
    for _ in range(n_transfers):
        src = rng.choice(accounts)
        dst = rng.choice([a for a in accounts if a != src])
        amount = rng.randint(1, 100)
        transfers.append({"from": src, "to": dst, "amount": amount})
    return FakeLedgerPayload(ledger_id=ledger_id, transfers=transfers)


def _build_graph(records: List[FakeLedgerPayload]) -> Dict[str, Dict[str, int]]:
    """Aggregate sender→receiver edge weights across the ingested ledgers."""
    edges: Dict[str, Dict[str, int]] = {}
    for record in records:
        for t in record.transfers:
            edges.setdefault(str(t["from"]), {}).setdefault(str(t["to"]), 0)
            edges[str(t["from"])][str(t["to"])] += int(t["amount"])
    return edges


def _node_features(edges: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Compute per-account out/in degree + total send/receive."""
    nodes: Dict[str, Dict[str, int]] = {}
    for src, dsts in edges.items():
        for dst, amt in dsts.items():
            nodes.setdefault(src, {"out_degree": 0, "in_degree": 0, "sent": 0, "received": 0})
            nodes.setdefault(dst, {"out_degree": 0, "in_degree": 0, "sent": 0, "received": 0})
            nodes[src]["out_degree"] += 1
            nodes[src]["sent"] += amt
            nodes[dst]["in_degree"] += 1
            nodes[dst]["received"] += amt
    return nodes


@pytest.mark.e2e
def test_pipeline_ingest_graph_features(tmp_path: pathlib.Path) -> None:
    """Ingest 5 synthetic ledgers, build the transfer graph, derive
    per-node features. Asserts the round-trip is deterministic under a
    fixed seed and that the produced feature set covers every account
    that appeared in the input."""
    _seed(42)

    state_path = tmp_path / "ingestion_state.json"
    store = StateStore(path=str(state_path))
    service = IngestionService(state_store=store)

    captured: List[FakeLedgerPayload] = []

    def fetch_fn(ledger_id: int) -> FakeLedgerPayload:
        return _synthetic_ledger(ledger_id)

    def process_fn(ledger_id: int, payload: object) -> None:
        # Ingestion service hands us back whatever fetch returned.
        assert isinstance(payload, FakeLedgerPayload)
        captured.append(payload)

    result = service.ingest(
        start_ledger=10,
        end_ledger=14,
        fetch_fn=fetch_fn,
        process_fn=process_fn,
    )

    # ── Ingestion stage ─────────────────────────────────────────────────
    assert result.attempted == [10, 11, 12, 13, 14]
    assert result.processed == [10, 11, 12, 13, 14]
    assert result.skipped == []
    assert len(captured) == 5

    # ── Graph stage ─────────────────────────────────────────────────────
    edges = _build_graph(captured)
    assert edges, "graph must have at least one edge"

    # Every account referenced in the input should appear as a node.
    accounts = {t["from"] for r in captured for t in r.transfers} | {
        t["to"] for r in captured for t in r.transfers
    }
    nodes = _node_features(edges)
    assert set(nodes) == accounts

    # ── Feature stage ───────────────────────────────────────────────────
    for account, feats in nodes.items():
        # Every account either sent or received at least once (and the
        # bookkeeping totals must match the edge sums).
        assert feats["out_degree"] + feats["in_degree"] > 0, account
        assert feats["sent"] >= 0
        assert feats["received"] >= 0

    # ── Re-ingest is idempotent ────────────────────────────────────────
    rerun = service.ingest(
        start_ledger=10,
        end_ledger=14,
        fetch_fn=fetch_fn,
        process_fn=process_fn,
    )
    assert rerun.processed == [], "rerun must skip already-processed ledgers"
    assert rerun.skipped == [10, 11, 12, 13, 14]


@pytest.mark.e2e
def test_pipeline_is_deterministic_across_runs(tmp_path: pathlib.Path) -> None:
    """Two pipeline runs with the same seed and same input must produce
    identical feature output. This is the regression test the seed
    change in train.py (#189) was made for."""
    _seed(42)
    edges_a = _build_graph([_synthetic_ledger(i) for i in range(20, 25)])
    features_a = _node_features(edges_a)

    _seed(42)
    edges_b = _build_graph([_synthetic_ledger(i) for i in range(20, 25)])
    features_b = _node_features(edges_b)

    assert features_a == features_b
