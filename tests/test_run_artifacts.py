"""Tests for astroml.artifacts.run_artifacts (issues #1216, #1219, #1220, #1223).

Standard library only: these run in the v1 workflow without torch or any
optional service.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from astroml.artifacts import run_artifacts as ra
from astroml.artifacts.run_artifacts import (
    METRICS_FILE,
    RUN_FILE,
    SCHEMA_VERSION,
    InvalidRunInputError,
    RunArtifactError,
    RunConflictError,
    atomic_write_json,
    compute_fingerprint,
    read_run_artifacts,
    remove_stale_staging,
    write_run_artifacts,
)

CONFIG = {"model": "gcn", "hidden_dim": 16, "lr": 0.01}
INPUTS = {"dataset": "Cora", "snapshot_sha256": "abc123"}
METRICS = {"val": {"accuracy": 0.81}, "test": {"accuracy": 0.79}, "epochs": 200}
WHEN = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _write(root: Path, run_id: str = "run-1", **overrides):
    kwargs = dict(config=CONFIG, inputs=INPUTS, metrics=METRICS, created_at=WHEN)
    kwargs.update(overrides)
    return write_run_artifacts(root, run_id, **kwargs)


def _leftovers(root: Path) -> list[str]:
    """Hidden staging/temp entries that must not outlive a write."""
    return sorted(p.name for p in root.iterdir() if p.name.startswith("."))


# ── #1216: machine-readable metrics ──────────────────────────────────────────


class TestMetricsFile:
    def test_writes_self_contained_run_directory(self, tmp_path: Path) -> None:
        result = _write(tmp_path)

        assert result.path == tmp_path / "run-1"
        assert sorted(p.name for p in result.path.iterdir()) == [METRICS_FILE, RUN_FILE]

        run = json.loads(result.run_file.read_text(encoding="utf-8"))
        assert run["schema_version"] == SCHEMA_VERSION
        assert run["run_id"] == "run-1"
        assert run["config"] == CONFIG
        assert run["inputs"] == INPUTS
        assert run["fingerprint"] == compute_fingerprint(CONFIG, INPUTS)
        assert run["created_at"] == WHEN.isoformat()

    def test_metrics_file_is_strict_json_with_schema_version(self, tmp_path: Path) -> None:
        result = _write(tmp_path)

        # parse_constant fires only for the non-standard NaN/Infinity tokens.
        def reject(token: str):
            raise AssertionError(f"non-standard JSON token {token}")

        data = json.loads(result.metrics_file.read_text(encoding="utf-8"), parse_constant=reject)
        assert data == {
            "schema_version": SCHEMA_VERSION,
            "run_id": "run-1",
            "fingerprint": result.fingerprint,
            "metrics": METRICS,
        }

    def test_round_trip(self, tmp_path: Path) -> None:
        _write(tmp_path)
        loaded = read_run_artifacts(tmp_path, "run-1")
        assert loaded["metrics"]["metrics"] == METRICS
        assert loaded["run"]["config"] == CONFIG

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_metric_is_rejected_and_nothing_is_written(
        self, tmp_path: Path, bad: float
    ) -> None:
        with pytest.raises(InvalidRunInputError):
            _write(tmp_path, metrics={"loss": bad})

        assert list(tmp_path.iterdir()) == []

    def test_non_json_metric_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidRunInputError):
            _write(tmp_path, metrics={"model": object()})
        assert list(tmp_path.iterdir()) == []

    def test_read_rejects_incomplete_or_foreign_runs(self, tmp_path: Path) -> None:
        result = _write(tmp_path)

        result.metrics_file.unlink()
        with pytest.raises(RunArtifactError, match="incomplete"):
            read_run_artifacts(tmp_path, "run-1")

    def test_read_rejects_unknown_schema_version(self, tmp_path: Path) -> None:
        result = _write(tmp_path)
        data = json.loads(result.metrics_file.read_text(encoding="utf-8"))
        data["schema_version"] = SCHEMA_VERSION + 1
        result.metrics_file.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(RunArtifactError, match="schema_version"):
            read_run_artifacts(tmp_path, "run-1")

    def test_read_rejects_corrupt_json(self, tmp_path: Path) -> None:
        result = _write(tmp_path)
        result.metrics_file.write_text("{not json", encoding="utf-8")
        with pytest.raises(RunArtifactError, match="cannot read"):
            read_run_artifacts(tmp_path, "run-1")


# ── #1219: atomic writes ─────────────────────────────────────────────────────


class TestAtomicWrites:
    def test_success_leaves_no_temp_files(self, tmp_path: Path) -> None:
        _write(tmp_path)
        assert _leftovers(tmp_path) == []

    def test_failure_while_writing_metrics_publishes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_replace = os.replace

        def flaky_replace(src, dst):
            if str(dst).endswith(METRICS_FILE):
                raise OSError("disk full")
            return real_replace(src, dst)

        monkeypatch.setattr(ra.os, "replace", flaky_replace)

        with pytest.raises(OSError, match="disk full"):
            _write(tmp_path)

        # No half-written run, and the staging directory was cleaned up.
        assert list(tmp_path.iterdir()) == []

    def test_atomic_write_json_keeps_old_content_on_failure(self, tmp_path: Path) -> None:
        target = tmp_path / "result.json"
        atomic_write_json(target, {"v": 1})

        with pytest.raises(ValueError):
            atomic_write_json(target, {"v": float("nan")})

        assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["result.json"]

    def test_failed_overwrite_restores_previous_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path, metrics={"accuracy": 0.5})

        real_rename = os.rename

        def flaky_rename(src, dst):
            # Fail only the final "staging -> run dir" step of the replacement.
            if str(src).endswith(ra._STAGING_SUFFIX):
                raise OSError("rename failed")
            return real_rename(src, dst)

        monkeypatch.setattr(ra.os, "rename", flaky_rename)
        with pytest.raises(OSError, match="rename failed"):
            _write(tmp_path, metrics={"accuracy": 0.9}, overwrite=True)
        monkeypatch.undo()

        loaded = read_run_artifacts(tmp_path, "run-1")
        assert loaded["metrics"]["metrics"] == {"accuracy": 0.5}
        assert _leftovers(tmp_path) == []

    def test_remove_stale_staging_only_removes_interrupted_writes(self, tmp_path: Path) -> None:
        _write(tmp_path)
        stale = tmp_path / ".run-2.abc123.staging"
        stale.mkdir()
        (stale / RUN_FILE).write_text("{}", encoding="utf-8")

        removed = remove_stale_staging(tmp_path)

        assert removed == [stale]
        assert not stale.exists()
        assert read_run_artifacts(tmp_path, "run-1")  # completed run untouched


# ── #1220: conflicting run identifiers ───────────────────────────────────────


class TestRunIdConflicts:
    def test_same_id_different_config_conflicts_and_keeps_original(self, tmp_path: Path) -> None:
        first = _write(tmp_path)
        original_metrics = first.metrics_file.read_text(encoding="utf-8")

        with pytest.raises(RunConflictError) as exc_info:
            _write(tmp_path, config={**CONFIG, "lr": 0.5}, metrics={"accuracy": 0.1})

        err = exc_info.value
        assert err.run_id == "run-1"
        assert err.same_inputs is False
        assert err.existing_fingerprint == first.fingerprint
        assert err.new_fingerprint != first.fingerprint
        assert "DIFFERENT" in str(err)
        assert first.metrics_file.read_text(encoding="utf-8") == original_metrics
        assert _leftovers(tmp_path) == []

    def test_same_id_identical_inputs_is_reported_as_a_rerun(self, tmp_path: Path) -> None:
        _write(tmp_path)

        with pytest.raises(RunConflictError) as exc_info:
            _write(tmp_path)

        assert exc_info.value.same_inputs is True
        assert "re-run" in str(exc_info.value)

    def test_existing_directory_without_run_file_still_conflicts(self, tmp_path: Path) -> None:
        (tmp_path / "run-1").mkdir()

        with pytest.raises(RunConflictError) as exc_info:
            _write(tmp_path)

        assert exc_info.value.existing_fingerprint is None
        assert exc_info.value.same_inputs is None

    def test_overwrite_true_replaces_the_run(self, tmp_path: Path) -> None:
        _write(tmp_path, metrics={"accuracy": 0.5})
        _write(
            tmp_path,
            config={**CONFIG, "lr": 0.5},
            metrics={"accuracy": 0.9},
            overwrite=True,
        )

        loaded = read_run_artifacts(tmp_path, "run-1")
        assert loaded["metrics"]["metrics"] == {"accuracy": 0.9}
        assert loaded["run"]["config"]["lr"] == 0.5
        assert _leftovers(tmp_path) == []

    def test_distinct_run_ids_coexist(self, tmp_path: Path) -> None:
        _write(tmp_path, "run-a")
        _write(tmp_path, "run-b")
        assert read_run_artifacts(tmp_path, "run-a")["run"]["run_id"] == "run-a"
        assert read_run_artifacts(tmp_path, "run-b")["run"]["run_id"] == "run-b"

    def test_lost_race_is_reported_as_a_conflict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A writer that appears after our check must not be clobbered."""
        real_rename = os.rename

        def racing_rename(src, dst):
            if str(src).endswith(ra._STAGING_SUFFIX):
                os.mkdir(dst)  # another writer wins first
                Path(dst, "marker").write_text("other writer", encoding="utf-8")
                raise FileExistsError(dst)
            return real_rename(src, dst)

        monkeypatch.setattr(ra.os, "rename", racing_rename)
        with pytest.raises(RunConflictError):
            _write(tmp_path)
        monkeypatch.undo()

        assert (tmp_path / "run-1" / "marker").read_text(encoding="utf-8") == "other writer"
        assert _leftovers(tmp_path) == []

    @pytest.mark.parametrize(
        "bad_id", ["", ".", "..", "../escape", "a/b", "a\\b", ".hidden", "x" * 129, "has space"]
    )
    def test_unsafe_run_ids_are_rejected_before_writing(self, tmp_path: Path, bad_id: str) -> None:
        with pytest.raises(InvalidRunInputError):
            _write(tmp_path, bad_id)
        assert list(tmp_path.iterdir()) == []

    def test_fingerprint_ignores_key_order_but_not_values(self) -> None:
        a = compute_fingerprint({"x": 1, "y": 2}, {"d": "a"})
        b = compute_fingerprint({"y": 2, "x": 1}, {"d": "a"})
        assert a == b
        assert a != compute_fingerprint({"x": 1, "y": 3}, {"d": "a"})
        assert a != compute_fingerprint({"x": 1, "y": 2}, {"d": "b"})


class TestReproduceSavedRun:
    def test_saved_config_and_inputs_can_be_replayed_under_a_new_id(self, tmp_path: Path) -> None:
        original = _write(tmp_path)
        saved = read_run_artifacts(tmp_path, "run-1")
        config = saved["run"]["config"]
        inputs = saved["run"]["inputs"]
        fingerprint = saved["run"]["fingerprint"]

        assert compute_fingerprint(config, inputs) == fingerprint

        replay = _write(
            tmp_path,
            "run-1-replay",
            config=config,
            inputs=inputs,
            metrics=saved["metrics"]["metrics"],
        )
        assert replay.fingerprint == original.fingerprint
        assert read_run_artifacts(tmp_path, "run-1-replay")["run"]["config"] == CONFIG

    def test_replaying_with_the_original_id_reports_a_rerun(self, tmp_path: Path) -> None:
        _write(tmp_path)
        saved = read_run_artifacts(tmp_path, "run-1")

        with pytest.raises(RunConflictError) as exc_info:
            _write(
                tmp_path,
                "run-1",
                config=saved["run"]["config"],
                inputs=saved["run"]["inputs"],
                metrics=saved["metrics"]["metrics"],
            )

        assert exc_info.value.same_inputs is True

    def test_changed_saved_inputs_are_detected_before_replay(self, tmp_path: Path) -> None:
        result = _write(tmp_path)
        data = json.loads(result.run_file.read_text(encoding="utf-8"))
        data["inputs"]["snapshot_sha256"] = "modified"
        result.run_file.write_text(json.dumps(data), encoding="utf-8")

        saved = read_run_artifacts(tmp_path, "run-1")

        assert (
            compute_fingerprint(saved["run"]["config"], saved["run"]["inputs"])
            != saved["run"]["fingerprint"]
        )
