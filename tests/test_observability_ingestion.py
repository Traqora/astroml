"""Tests for the ingestion heartbeat / stale-data check.

Covers the contract documented in ``docs/ingestion-monitoring.md``: the state
store stamps ``last_processed_at`` on every processed ledger, and the probe
grades that timestamp against the wall clock as OK / DEGRADED / FAIL.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from prometheus_client import REGISTRY

from astroml.ingestion.state import (
    DEFAULT_STATE_FILE,
    IngestionState,
    StateStore,
    resolve_state_path,
    utc_now_iso,
)
from astroml.observability.health import HealthStatus
from astroml.observability.ingestion import (
    DEFAULT_STALE_THRESHOLD_SECONDS,
    FAIL_THRESHOLD_MULTIPLIER,
    check_ingestion_heartbeat,
    parse_timestamp,
    refresh_ingestion_metrics,
    staleness_seconds,
    update_ingestion_metrics,
)

#: A fixed "now" so age assertions are exact rather than tolerance-based.
NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


def _iso(seconds_ago: float) -> str:
    """An ISO-8601 UTC timestamp ``seconds_ago`` before :data:`NOW`."""
    return (NOW - timedelta(seconds=seconds_ago)).isoformat()


class _FakeStore:
    """State store stand-in that returns a fixed state without touching disk."""

    def __init__(self, state: Any, path: str = "/tmp/fake_state.json") -> None:
        self.path = path
        self._state = state

    def load(self) -> Any:
        return self._state


class _BoomStore:
    """State store that raises on load, standing in for a corrupt state file."""

    path = "/tmp/corrupt_state.json"

    def load(self) -> Any:
        raise json.JSONDecodeError("Expecting value", "", 0)


def _state(**overrides: Any) -> IngestionState:
    values: dict[str, Any] = {
        "last_processed_ledger": 1_000_456,
        "last_processed_at": _iso(0),
    }
    values.update(overrides)
    return IngestionState(**values)


def _check(state: Any, **kwargs: Any) -> Any:
    return check_ingestion_heartbeat(_FakeStore(state), now=NOW, **kwargs)


class TestParseTimestamp:
    def test_reads_offset_timestamps(self) -> None:
        parsed = parse_timestamp("2026-09-24T12:00:00+00:00")

        assert parsed == NOW
        assert parsed is not None and parsed.tzinfo is not None

    def test_naive_timestamp_is_read_as_utc(self) -> None:
        assert parse_timestamp("2026-09-24T12:00:00") == NOW

    def test_other_offsets_are_normalised_to_utc(self) -> None:
        assert parse_timestamp("2026-09-24T14:00:00+02:00") == NOW

    @pytest.mark.parametrize("value", [None, "", "not-a-timestamp", "2026-13-45T99:99:99"])
    def test_unusable_values_are_none(self, value: str | None) -> None:
        assert parse_timestamp(value) is None


class TestStalenessSeconds:
    def test_computes_age_against_wall_clock(self) -> None:
        assert staleness_seconds(_iso(421.5), now=NOW) == pytest.approx(421.5)

    def test_accepts_unix_seconds(self) -> None:
        assert staleness_seconds(_iso(60), now=NOW.timestamp()) == pytest.approx(60.0)

    def test_a_naive_now_is_read_as_utc(self) -> None:
        naive = datetime(2026, 9, 24, 12, 0, 0)
        assert staleness_seconds(_iso(30), now=naive) == pytest.approx(30.0)

    def test_future_timestamp_is_clamped_to_zero(self) -> None:
        """A fast clock on the writer must not produce a negative age."""
        assert staleness_seconds(_iso(-90), now=NOW) == 0.0

    def test_no_timestamp_is_none(self) -> None:
        assert staleness_seconds(None, now=NOW) is None


class TestCheckIngestionHeartbeat:
    def test_fresh_ingestion_is_ok(self) -> None:
        result = _check(_state(last_processed_at=_iso(30)))

        assert result.status is HealthStatus.OK
        assert result.name == "ingestion"
        assert result.remediation == ""
        assert result.details["staleness_seconds"] == pytest.approx(30.0)
        assert result.details["last_processed_ledger"] == 1_000_456

    def test_stale_ingestion_is_degraded(self) -> None:
        result = _check(_state(last_processed_at=_iso(400)))

        assert result.status is HealthStatus.DEGRADED
        assert result.http_status == 200
        assert "past the 300s stale threshold" in result.remediation

    def test_very_stale_ingestion_fails(self) -> None:
        result = _check(_state(last_processed_at=_iso(700)))

        assert result.status is HealthStatus.FAIL
        assert result.http_status == 503
        assert "No ledger has been ingested for 700s" in result.remediation

    def test_threshold_boundaries_are_inclusive(self) -> None:
        """Exactly at the threshold escalates, mirroring the pool checks."""
        assert _check(_state(last_processed_at=_iso(300))).status is HealthStatus.DEGRADED
        assert _check(_state(last_processed_at=_iso(600))).status is HealthStatus.FAIL
        assert _check(_state(last_processed_at=_iso(299))).status is HealthStatus.OK

    def test_defaults_match_the_documented_thresholds(self) -> None:
        details = _check(_state()).details

        assert details["stale_threshold_seconds"] == DEFAULT_STALE_THRESHOLD_SECONDS
        assert details["fail_threshold_seconds"] == pytest.approx(
            DEFAULT_STALE_THRESHOLD_SECONDS * FAIL_THRESHOLD_MULTIPLIER
        )

    def test_missing_heartbeat_is_degraded_not_failed(self) -> None:
        result = _check(_state(last_processed_at=None))

        assert result.status is HealthStatus.DEGRADED
        assert result.details["staleness_seconds"] is None
        assert "No ingestion heartbeat is recorded" in result.remediation

    def test_a_legacy_state_file_reports_degraded(self, tmp_path: Any) -> None:
        """A file written before the field existed must load and degrade."""
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"last_processed_ledger": 7, "processed_ledgers": [7]}))

        result = check_ingestion_heartbeat(StateStore(str(path)), now=NOW)

        assert result.status is HealthStatus.DEGRADED
        assert result.details["last_processed_ledger"] == 7

    def test_unreadable_store_is_degraded_not_raised(self) -> None:
        result = check_ingestion_heartbeat(_BoomStore(), now=NOW)

        assert result.status is HealthStatus.DEGRADED
        assert result.details["error_type"] == "JSONDecodeError"
        assert "could not be read" in result.remediation

    def test_explicit_thresholds_are_honoured(self) -> None:
        result = _check(
            _state(last_processed_at=_iso(120)),
            stale_threshold_seconds=60,
            fail_threshold_seconds=240,
        )

        assert result.status is HealthStatus.DEGRADED
        assert result.details["fail_threshold_seconds"] == 240.0

    def test_env_overrides_apply_when_arguments_are_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INGESTION_STALE_THRESHOLD_SECONDS", "60")
        monkeypatch.setenv("INGESTION_FAIL_THRESHOLD_SECONDS", "120")

        result = _check(_state(last_processed_at=_iso(90)))

        assert result.status is HealthStatus.DEGRADED
        assert result.details["stale_threshold_seconds"] == 60.0
        assert result.details["fail_threshold_seconds"] == 120.0

    def test_non_numeric_env_override_falls_back_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INGESTION_STALE_THRESHOLD_SECONDS", "soon")

        details = _check(_state()).details

        assert details["stale_threshold_seconds"] == DEFAULT_STALE_THRESHOLD_SECONDS

    def test_details_expose_the_state_path_and_duration(self) -> None:
        result = _check(_state())

        assert result.details["path"] == "/tmp/fake_state.json"
        assert result.duration_ms >= 0.0


class TestFreshnessMetrics:
    def test_gauges_carry_the_timestamp_and_age(self) -> None:
        result = _check(_state(last_processed_at=_iso(120)))

        update_ingestion_metrics(result)

        assert REGISTRY.get_sample_value("astroml_ingestion_staleness_seconds") == pytest.approx(
            120.0
        )
        assert REGISTRY.get_sample_value(
            "astroml_ingestion_last_success_timestamp_seconds"
        ) == pytest.approx((NOW - timedelta(seconds=120)).timestamp())

    def test_absent_heartbeat_exports_nan(self) -> None:
        """NaN keeps the series present but never satisfies a stale comparison."""
        result = check_ingestion_heartbeat(_FakeStore(_state(last_processed_at=None)), now=NOW)

        update_ingestion_metrics(result)

        value = REGISTRY.get_sample_value("astroml_ingestion_staleness_seconds")

        assert value is not None and value != value  # NaN is never equal to itself
        assert not value > 900  # the alert expression must not match

    def test_refresh_returns_the_result_it_published(self) -> None:
        result = refresh_ingestion_metrics(_FakeStore(_state(last_processed_at=_iso(30))), now=NOW)

        assert result.status is HealthStatus.OK
        assert REGISTRY.get_sample_value("astroml_ingestion_staleness_seconds") == pytest.approx(
            30.0
        )


class TestStateStoreHeartbeat:
    def test_mark_processed_stamps_the_heartbeat(self, tmp_path: Any) -> None:
        store = StateStore(str(tmp_path / "state.json"))

        before = datetime.now(timezone.utc)
        state = store.mark_processed(42)
        after = datetime.now(timezone.utc)

        stamped = parse_timestamp(state.last_processed_at)
        assert stamped is not None
        assert before <= stamped <= after
        assert state.last_processed_ledger == 42

    def test_the_heartbeat_survives_a_reload(self, tmp_path: Any) -> None:
        store = StateStore(str(tmp_path / "state.json"))
        written = store.mark_processed(42).last_processed_at

        assert store.load().last_processed_at == written

    def test_the_heartbeat_is_advanced_by_later_ledgers(self, tmp_path: Any) -> None:
        store = StateStore(str(tmp_path / "state.json"))
        first = store.mark_processed(1).last_processed_at
        second = store.mark_processed(2).last_processed_at

        assert first is not None and second is not None
        assert parse_timestamp(second) >= parse_timestamp(first)

    def test_a_legacy_file_has_no_heartbeat(self, tmp_path: Any) -> None:
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"last_processed_ledger": 3, "processed_ledgers": [[1, 3]]}))

        assert StateStore(str(path)).load().last_processed_at is None

    def test_round_trips_the_new_field(self) -> None:
        state = IngestionState(last_processed_ledger=9, last_processed_at=utc_now_iso())

        assert IngestionState.from_dict(state.to_dict()).last_processed_at == (
            state.last_processed_at
        )


class TestResolveStatePath:
    def test_explicit_path_wins(self) -> None:
        assert resolve_state_path("/tmp/explicit.json") == "/tmp/explicit.json"

    def test_env_override_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INGESTION_STATE_FILE", "/data/shared_state.json")

        assert resolve_state_path() == "/data/shared_state.json"
        assert StateStore().path == "/data/shared_state.json"

    def test_default_when_nothing_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INGESTION_STATE_FILE", raising=False)

        assert resolve_state_path() == DEFAULT_STATE_FILE


class TestIngestionServiceHeartbeat:
    """The batched flush must stamp the same heartbeat as mark_processed."""

    def test_ingest_stream_stamps_the_heartbeat(self, tmp_path: Any) -> None:
        from astroml.ingestion.service import IngestionService

        store = StateStore(str(tmp_path / "state.json"))
        service = IngestionService(store)

        before = datetime.now(timezone.utc)
        outcomes = list(service.ingest_stream(start_ledger=1, end_ledger=5, batch_size=2))
        after = datetime.now(timezone.utc)

        assert [outcome.status for _, outcome in outcomes] == ["processed"] * 5
        stamped = parse_timestamp(store.load().last_processed_at)
        assert stamped is not None
        assert before <= stamped <= after

    def test_the_probe_reports_fresh_data_after_a_flush(self, tmp_path: Any) -> None:
        from astroml.ingestion.service import IngestionService

        store = StateStore(str(tmp_path / "state.json"))
        list(IngestionService(store).ingest_stream(start_ledger=1, end_ledger=5, batch_size=2))

        result = check_ingestion_heartbeat(store)

        assert result.status is HealthStatus.OK
        assert result.details["last_processed_ledger"] == 5

    def test_a_skipped_only_run_does_not_refresh_the_heartbeat(self, tmp_path: Any) -> None:
        """Nothing was processed, so nothing should look fresher than it is."""
        from astroml.ingestion.service import IngestionService

        store = StateStore(str(tmp_path / "state.json"))
        service = IngestionService(store)
        list(service.ingest_stream(start_ledger=1, end_ledger=5, batch_size=2))
        first = store.load().last_processed_at

        outcomes = list(service.ingest_stream(start_ledger=1, end_ledger=5, batch_size=2))

        assert [outcome.status for _, outcome in outcomes] == ["skipped"] * 5
        assert store.load().last_processed_at == first

    def test_an_error_before_any_flush_leaves_no_heartbeat(self, tmp_path: Any) -> None:
        from astroml.ingestion.service import IngestionService

        store = StateStore(str(tmp_path / "state.json"))
        service = IngestionService(store)

        def _boom(_ledger_id: int, _payload: object) -> None:
            raise RuntimeError("upstream exploded")

        with pytest.raises(RuntimeError):
            list(
                service.ingest_stream(start_ledger=1, end_ledger=5, batch_size=2, process_fn=_boom)
            )

        assert store.load().last_processed_at is None
