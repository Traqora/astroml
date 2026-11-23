"""Tests for astroml.artifacts.run_artifacts (issues #1216, #1219, #1220).

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


# ── #1222: schema validation on read ─────────────────────────────────────────


def _tamper(path: Path, change) -> None:
    """Apply a change to a stored JSON document for negative tests."""
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestReadSchemaValidation:
    def test_valid_run_passes_full_schema_validation(self, tmp_path: Path) -> None:
        """A round-tripped run satisfies every schema rule."""
        result = _write(tmp_path)
        loaded = read_run_artifacts(tmp_path, "run-1")
        assert loaded["run"]["fingerprint"] == result.fingerprint
        assert loaded["metrics"]["metrics"] == METRICS

    @pytest.mark.parametrize("name", [RUN_FILE, METRICS_FILE])
    def test_missing_required_field_is_rejected(self, tmp_path: Path, name: str) -> None:
        """Either file missing its fingerprint is rejected."""
        result = _write(tmp_path)
        target = result.run_file if name == RUN_FILE else result.metrics_file
        _tamper(target, lambda data: data.pop("fingerprint"))

        with pytest.raises(RunArtifactError, match="missing required field 'fingerprint'"):
            read_run_artifacts(tmp_path, "run-1")

    @pytest.mark.parametrize(
        ("name", "field", "bad_value"),
        [
            (RUN_FILE, "config", []),
            (RUN_FILE, "inputs", "Cora"),
            (RUN_FILE, "artifacts", {"metrics": "other.json"}),
            (METRICS_FILE, "metrics", []),
        ],
    )
    def test_wrong_field_type_or_reference_is_rejected(
        self, tmp_path: Path, name: str, field: str, bad_value: object
    ) -> None:
        """Wrong types and wrong artifact references are rejected."""
        result = _write(tmp_path)
        target = result.run_file if name == RUN_FILE else result.metrics_file
        _tamper(target, lambda data: data.__setitem__(field, bad_value))

        with pytest.raises(RunArtifactError, match="schema|must be|artifacts"):
            read_run_artifacts(tmp_path, "run-1")

    def test_invalid_created_at_is_rejected(self, tmp_path: Path) -> None:
        """created_at must be an ISO-8601 timestamp."""
        result = _write(tmp_path)
        _tamper(result.run_file, lambda data: data.__setitem__("created_at", "yesterday"))

        with pytest.raises(RunArtifactError, match="ISO-8601"):
            read_run_artifacts(tmp_path, "run-1")

    def test_fingerprint_mismatch_between_files_is_rejected(self, tmp_path: Path) -> None:
        """The two files must describe the same config and inputs."""
        result = _write(tmp_path)
        _tamper(result.metrics_file, lambda data: data.__setitem__("fingerprint", "0" * 64))

        with pytest.raises(RunArtifactError, match="different fingerprints"):
            read_run_artifacts(tmp_path, "run-1")

    def test_tampered_config_is_rejected_by_recomputed_fingerprint(self, tmp_path: Path) -> None:
        """A stored fingerprint must match the stored config and inputs."""
        result = _write(tmp_path)
        _tamper(result.run_file, lambda data: data["config"].__setitem__("lr", 0.5))

        with pytest.raises(RunArtifactError, match="fingerprint does not match"):
            read_run_artifacts(tmp_path, "run-1")

    def test_non_finite_value_in_stored_document_is_rejected(self, tmp_path: Path) -> None:
        """Strict JSON parsing rejects NaN even when written by an external tool."""
        result = _write(tmp_path)
        result.metrics_file.write_text(
            result.metrics_file.read_text(encoding="utf-8").replace("0.81", "NaN"),
            encoding="utf-8",
        )

        with pytest.raises(RunArtifactError, match="cannot read|not strict JSON"):
            read_run_artifacts(tmp_path, "run-1")

    def test_extra_fields_are_allowed_within_schema_version_1(self, tmp_path: Path) -> None:
        """Unknown fields are forward-compatible within a schema version."""
        result = _write(tmp_path)
        _tamper(result.run_file, lambda data: data.__setitem__("tool_version", "1.2.3"))
        _tamper(result.metrics_file, lambda data: data.__setitem__("notes", "extra"))

        loaded = read_run_artifacts(tmp_path, "run-1")
        assert loaded["run"]["tool_version"] == "1.2.3"
        assert loaded["metrics"]["notes"] == "extra"


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
