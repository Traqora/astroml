"""Integration coverage for the ingestion-to-snapshot fixture."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from astroml.db.schema import NormalizedTransaction
from astroml.pipeline.testing.fixtures import IngestionToSnapshotFixture


def _payment(operation_id: int, timestamp: datetime) -> dict:
    return {
        "id": operation_id,
        "type": "payment",
        "source_account": "G" + "A" * 55,
        "to": "G" + "B" * 55,
        "amount": "100.0",
        "asset_type": "native",
        "created_at": timestamp.isoformat(),
        "transaction_hash": f"{operation_id:064x}",
    }


def test_ingestion_payload_produces_snapshot(
    ingestion_to_snapshot_fixture: IngestionToSnapshotFixture,
) -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    payloads = [
        _payment(0x100000001, base),
        _payment(0x100000002, base + timedelta(minutes=30)),
    ]

    touched = ingestion_to_snapshot_fixture.ingest_payloads(payloads)
    assert touched == 2

    # Replaying the same payloads must update/retain, not duplicate, rows.
    assert ingestion_to_snapshot_fixture.ingest_payloads(payloads) == 2
    assert (
        ingestion_to_snapshot_fixture.session.query(NormalizedTransaction).count()
        == 2
    )

    snapshots = ingestion_to_snapshot_fixture.build_snapshots(
        window="1h",
        t0=base,
        t_now=base + timedelta(hours=1),
    )
    assert len(snapshots) == 1
    assert len(snapshots[0].edges) == 2
    assert snapshots[0].nodes == {"G" + "A" * 55, "G" + "B" * 55}


def test_ingestion_payload_rejects_malformed_record(
    ingestion_to_snapshot_fixture: IngestionToSnapshotFixture,
) -> None:
    malformed = {
        "type": "payment",
        "source_account": "G" + "A" * 55,
        # Missing destination, amount, timestamp, and transaction hash.
    }

    with pytest.raises(ValueError, match="Invalid ingestion payload"):
        ingestion_to_snapshot_fixture.ingest_payloads([malformed])
