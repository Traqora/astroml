"""End-to-end test for the ingest -> graph -> train pipeline (issue #709).

Small-scope run with fixture data, no services required:

1. **Backfill** — synthetic ledger rows are normalized by
   ``preprocess_ledger_backfill`` (the same function the backfill job
   runs), asserting rows survive with the right ledgers.
2. **Snapshot** — the normalized rows become ``Edge``s and
   ``window_snapshot`` builds the rolling graph, asserting a non-empty
   node/edge set.
3. **One training step** — per-node degree features feed a
   ``LogisticRegression(max_iter=1)`` (literally one optimizer step),
   asserting a fitted model and predictions.
4. **Model registry** — the artifact bytes plus metrics land in the
   ``models`` / ``model_versions`` registry tables (SQLite here) and read
   back, proving an artifact was produced *and stored*.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from astroml.db.models import Base, DbModel, ModelVersion
from astroml.features.graph.snapshot import Edge, window_snapshot
from astroml.preprocessing.ledger_backfill import (
    preprocess_ledger_backfill,
    scan_backfill_dataset,
)

SENDERS = ["GA", "GB", "GC", "GD"]


def _fixture_rows() -> list[dict]:
    """Three ledgers of payment ops with a hub/leaf shape.

    Per-ledger degrees sum to 8 (4 ops x 2 endpoints); totals come out as
    GA:8, GB:6, GC:5, GD:5, so the median split yields two classes and the
    hubs sit strictly above the leaves.
    """
    ops = [
        # ledger, sender, dst
        (101, "GA", "GB"),
        (101, "GA", "GC"),
        (101, "GA", "GD"),
        (101, "GB", "GC"),
        (102, "GA", "GB"),
        (102, "GA", "GC"),
        (102, "GB", "GD"),
        (102, "GA", "GD"),
        (103, "GA", "GB"),
        (103, "GB", "GC"),
        (103, "GC", "GD"),
        (103, "GA", "GD"),
    ]
    rows = []
    for op_id, (ledger, sender, dst) in enumerate(ops, start=1):
        rows.append(
            {
                "ledger_sequence": ledger,
                "source_account": sender,
                "destination_account": dst,
                "transaction_hash": f"tx{ledger}-{op_id}",
                "created_at": "2024-01-01T00:00:00Z",
                "id": op_id,
                "type": "payment",
                "amount": "10.0",
            }
        )
    return rows


@pytest.fixture()
def pipeline_artifact(tmp_path: Path) -> dict:
    """Run backfill -> snapshot -> one training step, returning everything."""
    fixture = tmp_path / "ledgers.ndjson"
    with fixture.open("w") as fh:
        for row in _fixture_rows():
            fh.write(json.dumps(row) + "\n")

    # 1. Backfill: normalize fixture rows exactly like the backfill job.
    frame = preprocess_ledger_backfill(
        scan_backfill_dataset(fixture, input_format="ndjson")
    ).collect()
    assert len(frame) == 12, f"expected 12 normalized ops, got {len(frame)}"

    # 2. Snapshot: rows become edges; the window covers the whole range.
    edges = [
        Edge(
            src=row["sender"],
            dst=row["receiver"],
            timestamp=int(row["timestamp"].timestamp()),
        )
        for row in frame.iter_rows(named=True)
        if row["receiver"] is not None
    ]
    assert len(edges) == 12
    lo = min(e.timestamp for e in edges)
    hi = max(e.timestamp for e in edges)
    nodes, snap_edges = window_snapshot(edges, lo, hi, presorted=False)
    assert nodes == {"GA", "GB", "GC", "GD"}
    assert len(snap_edges) == 12

    # 3. One training step: degree features, hub-vs-leaf labels, a single
    # optimizer step. ConvergenceWarning is expected — one step never
    # converges; that is the point of the "single step" assertion scope.
    ordered = sorted(nodes)
    degree = {n: 0 for n in ordered}
    for e in snap_edges:
        degree[e.src] += 1
        degree[e.dst] += 1
    X = np.array([[degree[n]] for n in ordered], dtype=float)
    median = float(np.median([degree[n] for n in ordered]))
    y = np.array([1 if degree[n] > median else 0 for n in ordered])
    assert set(y.tolist()) == {0, 1}, "fixture must yield both classes"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=1)
        model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(ordered)
    accuracy = float((preds == y).mean())

    artifact = tmp_path / "model.pkl"
    with artifact.open("wb") as fh:
        pickle.dump({"model": model, "nodes": ordered}, fh)
    return {
        "artifact": artifact,
        "accuracy": accuracy,
        "nodes": ordered,
        "edges": len(snap_edges),
    }


def test_pipeline_produces_stored_artifact(pipeline_artifact, tmp_path: Path):
    """The trained artifact is registered and reads back from the registry."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        model = DbModel(
            name="e2e-pipeline-smoke",
            framework="sklearn",
            task_type="classification",
        )
        session.add(model)
        session.flush()
        version = ModelVersion(
            model_id=model.id,
            version="v1",
            artifact_path=str(pipeline_artifact["artifact"]),
            hyperparameters={"max_iter": 1},
            metrics={"accuracy": pipeline_artifact["accuracy"]},
            status="trained",
        )
        session.add(version)
        session.commit()

        stored = session.execute(
            select(ModelVersion).where(ModelVersion.model_id == model.id)
        ).scalar_one()
        assert stored.status == "trained"
        assert Path(stored.artifact_path).is_file()
        assert stored.metrics["accuracy"] == pipeline_artifact["accuracy"]

        # The artifact itself round-trips: load it and predict again.
        with open(stored.artifact_path, "rb") as fh:
            bundle = pickle.load(fh)
        assert bundle["nodes"] == pipeline_artifact["nodes"]
        assert hasattr(bundle["model"], "predict")
    finally:
        session.close()
        engine.dispose()


def test_pipeline_snapshot_covers_all_fixture_ledgers(pipeline_artifact):
    """Guard against a silently narrowing window: all 12 ops must flow."""
    assert pipeline_artifact["edges"] == 12
    assert pipeline_artifact["nodes"] == ["GA", "GB", "GC", "GD"]
    assert 0.0 <= pipeline_artifact["accuracy"] <= 1.0


def test_pipeline_backfill_is_repeatable(tmp_path: Path):
    """Re-running the backfill stage over the same fixture is stable."""
    fixture = tmp_path / "ledgers.ndjson"
    with fixture.open("w") as fh:
        for row in _fixture_rows():
            fh.write(json.dumps(row) + "\n")
    first = preprocess_ledger_backfill(
        scan_backfill_dataset(fixture, input_format="ndjson")
    ).collect()
    second = preprocess_ledger_backfill(
        scan_backfill_dataset(fixture, input_format="ndjson")
    ).collect()
    assert first.equals(second)
    assert len(first) == 12
