"""Tests for ingestion heartbeat / stale-data alerts (Issue #758)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from astroml.ingestion.state import StateStore
from astroml.observability.health import HealthStatus
from astroml.observability.ingestion import check_ingestion_heartbeat


class TestCheckIngestionHeartbeat:
    def test_ok_when_recent_ingestion(self, tmp_path: Path) -> None:
        state_path = tmp_path / "ingestion_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "last_processed_ledger": 1000,
                    "processed_ledgers": [1000],
                    "last_processed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            ),
            encoding="utf-8",
        )
        store = StateStore(str(state_path))

        result = check_ingestion_heartbeat(store, stale_threshold_seconds=300)

        assert result.status is HealthStatus.OK
        assert result.details["last_processed_ledger"] == 1000
        assert result.remediation == ""

    def test_degraded_when_stale(self, tmp_path: Path) -> None:
        state_path = tmp_path / "ingestion_state.json"
        stale_at = datetime.now(timezone.utc) - timedelta(seconds=400)
        state_path.write_text(
            json.dumps(
                {
                    "last_processed_ledger": 1000,
                    "processed_ledgers": [1000],
                    "last_processed_at": stale_at.isoformat().replace("+00:00", "Z"),
                }
            ),
            encoding="utf-8",
        )
        store = StateStore(str(state_path))

        result = check_ingestion_heartbeat(store, stale_threshold_seconds=300)

        assert result.status is HealthStatus.DEGRADED
        assert "stale threshold" in result.remediation

    def test_fail_when_critically_stale(self, tmp_path: Path) -> None:
        state_path = tmp_path / "ingestion_state.json"
        stale_at = datetime.now(timezone.utc) - timedelta(seconds=700)
        state_path.write_text(
            json.dumps(
                {
                    "last_processed_ledger": 1000,
                    "processed_ledgers": [1000],
                    "last_processed_at": stale_at.isoformat().replace("+00:00", "Z"),
                }
            ),
            encoding="utf-8",
        )
        store = StateStore(str(state_path))

        result = check_ingestion_heartbeat(store, stale_threshold_seconds=300)

        assert result.status is HealthStatus.FAIL
        assert "fail threshold" in result.remediation

    def test_degraded_when_no_timestamp_recorded(self, tmp_path: Path) -> None:
        state_path = tmp_path / "ingestion_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "last_processed_ledger": 1000,
                    "processed_ledgers": [1000],
                }
            ),
            encoding="utf-8",
        )
        store = StateStore(str(state_path))

        result = check_ingestion_heartbeat(store)

        assert result.status is HealthStatus.DEGRADED
        assert "No ingestion timestamp" in result.remediation

    def test_state_store_records_timestamp_on_mark_processed(self, tmp_path: Path) -> None:
        state_path = tmp_path / "ingestion_state.json"
        store = StateStore(str(state_path))

        store.mark_processed(1000)
        state = store.load()

        assert state.last_processed_at is not None
        assert state.last_processed_at.endswith("Z")

    def test_state_store_round_trip_preserves_timestamp(self, tmp_path: Path) -> None:
        state_path = tmp_path / "ingestion_state.json"
        store = StateStore(str(state_path))
        store.mark_processed(1000)

        reloaded = StateStore(str(state_path)).load()

        assert reloaded.last_processed_at == store.load().last_processed_at
