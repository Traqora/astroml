"""Self-contained, atomically written run artifacts for the v1 workflow.

A *run* is stored as one directory named after its run id::

    <root>/<run_id>/
        run.json       run id, config, input identifiers, fingerprint, timestamp
        metrics.json   machine-readable metrics for the run

Guarantees (see ``docs/run-artifacts.md``):

* **Atomic (#1219).** Files are written to a hidden staging directory and the
  directory is renamed into place only once every file is complete and
  fsynced. A reader sees either the whole run or no run at all, never a
  half-written one. A crash leaves at most a ``.<run_id>.*.staging`` directory,
  which :func:`remove_stale_staging` clears.
* **No silent overwrite (#1220).** Writing a run id that already exists raises
  :class:`RunConflictError` and leaves the existing run untouched, whether or
  not the config matches. ``overwrite=True`` replaces it explicitly, and a
  failed replacement restores the previous run.
* **Machine-readable metrics (#1216).** ``metrics.json`` is strict JSON
  (``NaN``/``Infinity`` are rejected, not written as invalid JSON) with a
  ``schema_version`` so consumers can detect format changes.
* **Validated reads (#1222).** :func:`read_run_artifacts` checks the required
  fields and types in both files and verifies that their fingerprints agree
  with each other and with ``config`` + ``inputs``.

Local filesystem only; it uses the standard library, so it works in the v1
workflow without torch, cloud storage or any other optional service.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Mapping, cast

from astroml.utils.exceptions import AstroMLError

__all__ = [
    "SCHEMA_VERSION",
    "RUN_FILE",
    "METRICS_FILE",
    "RunArtifactError",
    "InvalidRunInputError",
    "RunConflictError",
    "RunArtifacts",
    "validate_run_id",
    "compute_fingerprint",
    "atomic_write_json",
    "write_run_artifacts",
    "read_run_artifacts",
    "remove_stale_staging",
]

#: Version of the ``run.json`` / ``metrics.json`` layout. Bump on breaking changes.
SCHEMA_VERSION = 1
RUN_FILE = "run.json"
METRICS_FILE = "metrics.json"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_STAGING_SUFFIX = ".staging"
_REPLACED_SUFFIX = ".replaced"


class RunArtifactError(AstroMLError):
    """A run's artifacts could not be written or read."""

    code: ClassVar[str] = "RUN_ARTIFACT_ERROR"
    public_message: ClassVar[str] = "Run artifacts could not be processed."


class InvalidRunInputError(RunArtifactError, ValueError):
    """The run id, config, inputs or metrics are not valid for persistence."""

    code: ClassVar[str] = "RUN_ARTIFACT_INVALID_INPUT"
    public_message: ClassVar[str] = "The run artifacts request was invalid."


class RunConflictError(RunArtifactError):
    """A run with this id already exists, so it was not overwritten.

    Attributes:
        run_id: The conflicting run id.
        path: Directory of the existing run.
        existing_fingerprint: Fingerprint stored with the existing run, or
            ``None`` if it could not be read.
        new_fingerprint: Fingerprint of the run that was being written.
        same_inputs: ``True`` when the existing run has the same config and
            inputs (a re-run), ``False`` when they differ (an id collision),
            ``None`` when that cannot be determined.
    """

    code: ClassVar[str] = "RUN_ID_CONFLICT"
    public_message: ClassVar[str] = "A run with this identifier already exists."

    def __init__(
        self,
        run_id: str,
        path: Path,
        existing_fingerprint: str | None,
        new_fingerprint: str,
    ) -> None:
        if existing_fingerprint is None:
            same: bool | None = None
            detail = "its existing fingerprint could not be read"
        elif existing_fingerprint == new_fingerprint:
            same = True
            detail = "the existing run has identical config and inputs (a re-run)"
        else:
            same = False
            detail = (
                "the existing run has DIFFERENT config or inputs "
                f"(existing {existing_fingerprint[:12]}, new {new_fingerprint[:12]})"
            )
        super().__init__(
            f"Run id {run_id!r} already exists at {path}: {detail}. "
            "Pick a new run id or pass overwrite=True to replace it.",
            run_id=run_id,
            path=str(path),
        )
        self.run_id = run_id
        self.path = path
        self.existing_fingerprint = existing_fingerprint
        self.new_fingerprint = new_fingerprint
        self.same_inputs = same


@dataclass(frozen=True)
class RunArtifacts:
    """Where a run was written."""

    run_id: str
    path: Path
    run_file: Path
    metrics_file: Path
    fingerprint: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` if it is a safe directory name, else raise.

    Allowed: 1-128 characters from ``A-Z a-z 0-9 . _ -``, starting with an
    alphanumeric character. This rules out path separators, ``.``/``..`` and
    hidden names (which are reserved for staging directories).
    """
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise InvalidRunInputError(
            f"Invalid run id {run_id!r}: use 1-128 characters from A-Z a-z 0-9 . _ - "
            "and start with a letter or digit."
        )
    return run_id


def _canonical_json(obj: Any) -> str:
    try:
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
        )
    except (TypeError, ValueError) as exc:
        raise InvalidRunInputError(
            f"Value is not strict-JSON serialisable (NaN/Infinity and non-JSON types "
            f"are rejected): {exc}"
        ) from exc


def compute_fingerprint(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of ``config`` and ``inputs``.

    Independent of key order, so two runs with the same settings and the same
    input identifiers always share a fingerprint.
    """
    return hashlib.sha256(
        _canonical_json({"config": config, "inputs": inputs}).encode("utf-8")
    ).hexdigest()


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync (not supported on every platform)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> Path:
    """Write ``obj`` as strict JSON to ``path`` atomically.

    The data goes to a temp file in the same directory, is flushed and
    fsynced, and is then moved over ``path`` with :func:`os.replace`. If
    anything fails the destination is left exactly as it was and the temp file
    is removed.
    """
    target = Path(path)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    _fsync_dir(target.parent)
    return target


def _read_fingerprint(run_dir: Path) -> str | None:
    try:
        with open(run_dir / RUN_FILE, "r", encoding="utf-8") as handle:
            value = json.load(handle).get("fingerprint")
        return value if isinstance(value, str) else None
    except (OSError, ValueError, AttributeError):
        return None


def _reject_json_constant(token: str) -> None:
    """Reject non-standard JSON tokens such as ``NaN`` and ``Infinity``."""
    raise ValueError(f"non-standard JSON token {token}")


def _load_document(path: Path, name: str, run_id: str) -> dict[str, Any]:
    """Read one artifact file as a strict-JSON object with the common fields checked."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=_reject_json_constant)
    except FileNotFoundError as exc:
        raise RunArtifactError(
            f"Run {run_id!r} is incomplete: {name} is missing in {path.parent}."
        ) from exc
    except (OSError, ValueError) as exc:
        raise RunArtifactError(f"Run {run_id!r}: cannot read {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RunArtifactError(f"Run {run_id!r}: {name} must be a JSON object.")
    if type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise RunArtifactError(
            f"Run {run_id!r}: {name} has unsupported schema_version "
            f"{data.get('schema_version')!r} (expected {SCHEMA_VERSION})."
        )
    if data.get("run_id") != run_id:
        raise RunArtifactError(f"Run {run_id!r}: {name} belongs to run {data.get('run_id')!r}.")
    try:
        _canonical_json(data)
    except InvalidRunInputError as exc:
        raise RunArtifactError(f"Run {run_id!r}: {name} is not strict JSON: {exc}") from exc
    return data


def _require(document: Mapping[str, Any], key: str, expected: type, name: str, run_id: str) -> Any:
    """Return a required field of the expected type or raise a schema error."""
    if key not in document:
        raise RunArtifactError(f"Run {run_id!r}: {name} is missing required field {key!r}.")
    value = document[key]
    if expected is int:
        valid = type(value) is int
    elif expected is dict:
        valid = isinstance(value, dict)
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise RunArtifactError(
            f"Run {run_id!r}: {name} field {key!r} must be {expected.__name__}, "
            f"got {type(value).__name__}."
        )
    return value


def _validate_document_schema(document: Mapping[str, Any], name: str, run_id: str) -> str:
    """Validate the fields of one artifact document and return its fingerprint."""
    fingerprint = cast(str, _require(document, "fingerprint", str, name, run_id))
    if not _FINGERPRINT_RE.fullmatch(fingerprint):
        raise RunArtifactError(
            f"Run {run_id!r}: {name} field 'fingerprint' must be a lowercase SHA-256 hex digest."
        )

    if name == RUN_FILE:
        created_at = _require(document, "created_at", str, name, run_id)
        try:
            datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise RunArtifactError(
                f"Run {run_id!r}: {name} field 'created_at' is not an ISO-8601 timestamp."
            ) from exc
        _require(document, "config", dict, name, run_id)
        _require(document, "inputs", dict, name, run_id)
        artifacts = _require(document, "artifacts", dict, name, run_id)
        if artifacts.get("metrics") != METRICS_FILE:
            raise RunArtifactError(
                f"Run {run_id!r}: {name} field 'artifacts' must list {METRICS_FILE!r}."
            )
        return fingerprint

    _require(document, "metrics", dict, name, run_id)
    return fingerprint


def _validate_run_relationships(
    run_document: Mapping[str, Any],
    metrics_document: Mapping[str, Any],
    run_fingerprint: str,
    metrics_fingerprint: str,
) -> None:
    """Check that both files describe the same config and inputs."""
    if run_fingerprint != metrics_fingerprint:
        raise RunArtifactError(
            f"Run {run_document['run_id']!r}: {RUN_FILE} and {METRICS_FILE} have "
            "different fingerprints."
        )
    try:
        expected = compute_fingerprint(run_document["config"], run_document["inputs"])
    except InvalidRunInputError as exc:
        raise RunArtifactError(
            f"Run {run_document['run_id']!r}: config or inputs are not strict JSON: {exc}"
        ) from exc
    if run_fingerprint != expected:
        raise RunArtifactError(
            f"Run {run_document['run_id']!r}: fingerprint does not match config and inputs."
        )


# ── Public API ───────────────────────────────────────────────────────────────


def write_run_artifacts(
    root: str | os.PathLike[str],
    run_id: str,
    *,
    config: Mapping[str, Any],
    inputs: Mapping[str, Any],
    metrics: Mapping[str, Any],
    overwrite: bool = False,
    created_at: datetime | None = None,
) -> RunArtifacts:
    """Write ``run.json`` and ``metrics.json`` for ``run_id`` atomically.

    Args:
        root: Directory that holds all runs (created if missing).
        run_id: Unique run identifier; see :func:`validate_run_id`.
        config: Hyper-parameters / settings that produced the run.
        inputs: Identifiers of the inputs (dataset name, snapshot hash, ...).
        metrics: Result metrics. Must be strict JSON: finite numbers only.
        overwrite: Replace an existing run with the same id instead of raising.
        created_at: Timestamp to record (UTC now by default).

    Raises:
        InvalidRunInputError: bad run id, or non-serialisable / non-finite values.
            Nothing is written.
        RunConflictError: ``run_id`` already exists and ``overwrite`` is false.
            The existing run is untouched.
    """
    validate_run_id(run_id)
    # Validate everything before touching the filesystem.
    fingerprint = compute_fingerprint(config, inputs)
    _canonical_json(metrics)

    created = (created_at or datetime.now(timezone.utc)).isoformat()
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    final = root_path / run_id

    if os.path.lexists(final) and not overwrite:
        raise RunConflictError(run_id, final, _read_fingerprint(final), fingerprint)

    stage = Path(tempfile.mkdtemp(dir=str(root_path), prefix=f".{run_id}.", suffix=_STAGING_SUFFIX))
    try:
        atomic_write_json(
            stage / RUN_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "created_at": created,
                "fingerprint": fingerprint,
                "config": dict(config),
                "inputs": dict(inputs),
                "artifacts": {"metrics": METRICS_FILE},
            },
        )
        atomic_write_json(
            stage / METRICS_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "fingerprint": fingerprint,
                "metrics": dict(metrics),
            },
        )
        _fsync_dir(stage)
        _publish(stage, final, run_id, fingerprint, overwrite)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    _fsync_dir(root_path)

    return RunArtifacts(
        run_id=run_id,
        path=final,
        run_file=final / RUN_FILE,
        metrics_file=final / METRICS_FILE,
        fingerprint=fingerprint,
    )


def _publish(stage: Path, final: Path, run_id: str, fingerprint: str, overwrite: bool) -> None:
    """Move the completed staging directory into place."""
    if not os.path.lexists(final):
        try:
            os.rename(stage, final)
        except OSError:
            # Lost a race with a concurrent writer of the same id.
            if os.path.lexists(final):
                raise RunConflictError(
                    run_id, final, _read_fingerprint(final), fingerprint
                ) from None
            raise
        return

    if not overwrite:  # appeared between the check and now
        raise RunConflictError(run_id, final, _read_fingerprint(final), fingerprint)

    backup = Path(
        tempfile.mkdtemp(dir=str(final.parent), prefix=f".{run_id}.", suffix=_REPLACED_SUFFIX)
    )
    os.rmdir(backup)  # only reserved a unique name
    os.rename(final, backup)
    try:
        os.rename(stage, final)
    except BaseException:
        os.rename(backup, final)  # restore the previous run
        raise
    shutil.rmtree(backup, ignore_errors=True)


def read_run_artifacts(root: str | os.PathLike[str], run_id: str) -> dict[str, Any]:
    """Load and fully validate a run written by :func:`write_run_artifacts`.

    Both files must be strict-JSON objects with ``schema_version``, ``run_id``
    and ``fingerprint``. ``run.json`` must additionally contain ``created_at``
    (ISO-8601), ``config`` and ``inputs`` objects and an ``artifacts`` object
    naming ``metrics.json``; ``metrics.json`` must contain a ``metrics`` object.
    Extra fields are allowed, but the two fingerprints must match each other
    and the SHA-256 of the canonical JSON of ``config`` + ``inputs``.

    Returns:
        ``{"run": <run.json>, "metrics": <metrics.json>}``.

    Raises:
        InvalidRunInputError: ``run_id`` is not a safe directory name.
        RunArtifactError: a file is missing, corrupt or does not satisfy the
            v1 schema, or the fingerprint does not match its contents.
    """
    validate_run_id(run_id)
    run_dir = Path(root) / run_id
    documents = {
        key: _load_document(run_dir / name, name, run_id)
        for key, name in (("run", RUN_FILE), ("metrics", METRICS_FILE))
    }
    run_document = documents["run"]
    metrics_document = documents["metrics"]
    run_fingerprint = _validate_document_schema(run_document, RUN_FILE, run_id)
    metrics_fingerprint = _validate_document_schema(metrics_document, METRICS_FILE, run_id)
    _validate_run_relationships(
        run_document, metrics_document, run_fingerprint, metrics_fingerprint
    )
    return documents


def remove_stale_staging(root: str | os.PathLike[str]) -> list[Path]:
    """Delete leftover staging/backup directories from interrupted writes.

    Only call this when no writer is active on ``root``. Returns the removed
    paths. Completed runs are never touched.
    """
    root_path = Path(root)
    removed: list[Path] = []
    if not root_path.is_dir():
        return removed
    for child in root_path.iterdir():
        if (
            child.is_dir()
            and child.name.startswith(".")
            and child.name.endswith((_STAGING_SUFFIX, _REPLACED_SUFFIX))
        ):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child)
    return removed
