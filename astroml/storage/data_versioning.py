from astroml.utils.exceptions import AstroMLError
"""Data versioning integration with DVC for dataset reproducibility.

Provides versioned dataset management, snapshots, tagging, annotation,
and comparison backed by DVC remotes (S3, GCS, local).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatasetVersion:
    """A versioned dataset snapshot."""

    version_id: str
    name: str
    version: str
    path: str
    dvc_hash: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)
    parent_version_id: str | None = None
    size_bytes: int = 0
    num_files: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionDiff:
    """Difference between two dataset versions."""

    version_a: str
    version_b: str
    added_files: list[str] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    size_diff_bytes: int = 0
    summary: str = ""


class DataVersionControl:
    """DVC-backed data versioning manager.

    Provides dataset versioning, snapshot management, tagging, annotation,
    comparison, and remote storage operations.

    Usage::

        dvc = DataVersionControl(repo_root="/workspace")
        ver = dvc.add_dataset("training_data", "data/train.csv", "v1.0")
        dvc.tag_version(ver.version_id, ["production"])
        diff = dvc.compare_versions("v1.0", "v2.0")
    """

    def __init__(
        self,
        repo_root: str | Path = ".",
        remote: str = "origin",
        dvc_path: str = "dvc",
    ) -> None:
        """Initialize DVC-backed version control.

        Args:
            repo_root: Root of the DVC repository.
            remote: DVC remote name for push/pull.
            dvc_path: Path to the ``dvc`` executable.
        """
        self.repo_root = Path(repo_root)
        self.remote = remote
        self.dvc_path = dvc_path
        self._versions: dict[str, DatasetVersion] = {}
        self._enabled = self._check_dvc()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether DVC is available on this system."""
        return self._enabled

    def _check_dvc(self) -> bool:
        """Check whether DVC is installed and a DVC repo is initialized."""
        try:
            result = subprocess.run(
                [self.dvc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning("DVC not available: %s", result.stderr.strip())
                return False
            # Check if .dvc directory exists
            if not (self.repo_root / ".dvc").exists():
                logger.warning("Not a DVC repository (no .dvc directory)")
                return False
            logger.info("DVC detected: %s", result.stdout.strip())
            return True
        except FileNotFoundError:
            logger.warning("DVC executable not found at %s", self.dvc_path)
            return False
        except Exception as exc:
            logger.warning("DVC availability check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Dataset versioning
    # ------------------------------------------------------------------

    def add_dataset(
        self,
        name: str,
        path: str | Path,
        version: str = "latest",
        description: str = "",
        tags: list[str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> DatasetVersion:
        """Add and version a dataset with DVC tracking.

        Args:
            name: Dataset name.
            path: Path to dataset file or directory.
            version: Version label.
            description: Human-readable description.
            tags: Optional tags.
            annotations: Optional key-value annotations.

        Returns:
            The created DatasetVersion.
        """
        ver = DatasetVersion(
            version_id=uuid.uuid4().hex[:12],
            name=name,
            version=version,
            path=str(path),
            description=description,
            tags=tags or [],
            annotations=annotations or {},
        )

        # Track with DVC if available
        if self._enabled:
            try:
                result = subprocess.run(
                    [self.dvc_path, "add", str(path)],
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    # Extract hash from .dvc file
                    dvc_file = Path(str(path) + ".dvc")
                    if dvc_file.exists():
                        import yaml

                        with open(dvc_file) as f:
                            dvc_data = yaml.safe_load(f)
                        ver.dvc_hash = (
                            dvc_data.get("outs", [{}])[0].get("md5", "")
                            if dvc_data.get("outs")
                            else ""
                        )
                    logger.info(
                        "Dataset tracked with DVC: %s (hash=%s)",
                        path,
                        ver.dvc_hash,
                    )
                else:
                    logger.warning("DVC add failed: %s", result.stderr.strip())
            except Exception as exc:
                logger.warning("DVC tracking failed: %s", exc)

        # Compute size and file count
        full_path = self.repo_root / path
        if full_path.exists():
            if full_path.is_file():
                ver.size_bytes = full_path.stat().st_size
                ver.num_files = 1
            elif full_path.is_dir():
                for root, _, files in os.walk(full_path):
                    for f in files:
                        fp = Path(root) / f
                        ver.size_bytes += fp.stat().st_size
                        ver.num_files += 1

        self._versions[ver.version_id] = ver
        return ver

    def get_version(self, version_id: str) -> DatasetVersion | None:
        """Get a dataset version by ID.

        Args:
            version_id: Version identifier.

        Returns:
            DatasetVersion or None.
        """
        return self._versions.get(version_id)

    def list_versions(
        self,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> list[DatasetVersion]:
        """List dataset versions with optional filtering.

        Args:
            name: Filter by dataset name.
            tags: Filter by tags (all must match).

        Returns:
            Filtered list of versions.
        """
        results = list(self._versions.values())
        if name:
            results = [v for v in results if v.name == name]
        if tags:
            tag_set = set(tags)
            results = [v for v in results if tag_set.issubset(set(v.tags))]
        return sorted(results, key=lambda v: v.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Tagging and annotations
    # ------------------------------------------------------------------

    def tag_version(
        self,
        version_id: str,
        tags: list[str],
    ) -> DatasetVersion:
        """Add tags to a dataset version.

        Args:
            version_id: Version to tag.
            tags: Tags to add.

        Returns:
            Updated version.
        """
        ver = self._get(version_id)
        for tag in tags:
            if tag not in ver.tags:
                ver.tags.append(tag)
        return ver

    def annotate(
        self,
        version_id: str,
        annotations: dict[str, str],
    ) -> DatasetVersion:
        """Add annotations to a dataset version.

        Args:
            version_id: Version to annotate.
            annotations: Key-value annotations.

        Returns:
            Updated version.
        """
        ver = self._get(version_id)
        ver.annotations.update(annotations)
        return ver

    # ------------------------------------------------------------------
    # Version comparison
    # ------------------------------------------------------------------

    def compare_versions(
        self,
        version_id_a: str,
        version_id_b: str,
    ) -> VersionDiff:
        """Compare two dataset versions.

        Args:
            version_id_a: First version.
            version_id_b: Second version.

        Returns:
            VersionDiff with file changes and size difference.
        """
        ver_a = self._get(version_id_a)
        ver_b = self._get(version_id_b)

        diff = VersionDiff(
            version_a=ver_a.version,
            version_b=ver_b.version,
            size_diff_bytes=ver_b.size_bytes - ver_a.size_bytes,
        )

        path_a = self.repo_root / ver_a.path
        path_b = self.repo_root / ver_b.path

        files_a: set[str] = set()
        files_b: set[str] = set()

        for p, fileset in [(path_a, files_a), (path_b, files_b)]:
            if p.is_file():
                fileset.add(p.name)
            elif p.is_dir():
                for root, _, filenames in os.walk(p):
                    for fn in filenames:
                        fileset.add(str(Path(root) / fn))
            else:
                logger.warning("Path %s does not exist", p)

        diff.added_files = sorted(files_b - files_a)
        diff.removed_files = sorted(files_a - files_b)

        # Modified files (same name, possibly different)
        common = files_a & files_b
        for fname in common:
            fa = path_a / fname if path_a.is_dir() else path_a
            fb = path_b / fname if path_b.is_dir() else path_b
            try:
                if fa.stat().st_size != fb.stat().st_size:
                    diff.modified_files.append(fname)
            except OSError:
                diff.modified_files.append(fname)

        diff.modified_files.sort()

        # Build summary
        parts = [
            f"Comparing {ver_a.version} → {ver_b.version}",
            f"{len(diff.added_files)} added",
            f"{len(diff.removed_files)} removed",
            f"{len(diff.modified_files)} modified",
            f"Size diff: {diff.size_diff_bytes:,} bytes",
        ]
        diff.summary = "; ".join(parts)

        return diff

    # ------------------------------------------------------------------
    # DVC operations
    # ------------------------------------------------------------------

    def push(
        self, version_id: str | None = None
    ) -> dict[str, Any]:
        """Push tracked data to the DVC remote.

        Args:
            version_id: Optional version to push; if None, pushes all.

        Returns:
            Dict with success status and output.
        """
        if not self._enabled:
            return {"success": False, "error": "DVC not available"}

        try:
            result = subprocess.run(
                [self.dvc_path, "push"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except Exception as exc:
            logger.error("DVC push failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def pull(self, version_id: str | None = None) -> dict[str, Any]:
        """Pull tracked data from the DVC remote.

        Args:
            version_id: Optional version to pull; if None, pulls all.

        Returns:
            Dict with success status and output.
        """
        if not self._enabled:
            return {"success": False, "error": "DVC not available"}

        try:
            result = subprocess.run(
                [self.dvc_path, "pull"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except Exception as exc:
            logger.error("DVC pull failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def status(self) -> dict[str, Any]:
        """Get DVC repository status.

        Returns:
            Dict with tracking status.
        """
        if not self._enabled:
            return {"enabled": False, "status": "DVC not available"}

        try:
            result = subprocess.run(
                [self.dvc_path, "status"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "enabled": True,
                "changed": result.returncode != 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip(),
            }
        except Exception as exc:
            return {"enabled": True, "error": str(exc)}

    def snapshot(self, version_id: str) -> dict[str, Any]:
        """Create a lightweight metadata snapshot of a version.

        Args:
            version_id: Version to snapshot.

        Returns:
            Dict with snapshot data.
        """
        ver = self._get(version_id)
        return {
            "version_id": ver.version_id,
            "name": ver.name,
            "version": ver.version,
            "path": ver.path,
            "dvc_hash": ver.dvc_hash,
            "size_bytes": ver.size_bytes,
            "num_files": ver.num_files,
            "tags": ver.tags,
            "annotations": ver.annotations,
            "created_at": ver.created_at,
        }

    def export_snapshot_json(
        self, version_id: str, output_path: str | Path | None = None
    ) -> str:
        """Export a version snapshot as JSON.

        Args:
            version_id: Version to export.
            output_path: Optional file path to write to.

        Returns:
            JSON string.
        """
        data = self.snapshot(version_id)
        json_str = json.dumps(data, indent=2, default=str)
        if output_path:
            Path(output_path).write_text(json_str)
        return json_str

    # ------------------------------------------------------------------
    # Pipeline reproducibility
    # ------------------------------------------------------------------

    def run_pipeline_stage(
        self,
        stage_name: str,
        command: str,
        dependencies: list[str],
        outputs: list[str],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a DVC pipeline stage for reproducibility.

        Args:
            stage_name: Name of the pipeline stage.
            command: Shell command to run.
            dependencies: List of dependency file paths.
            outputs: List of output file paths.
            params: Optional YAML params for the stage.

        Returns:
            Dict with execution status.
        """
        if not self._enabled:
            return {"success": False, "error": "DVC not available"}

        cmd = [
            self.dvc_path,
            "run",
            "-n",
            stage_name,
        ]
        for dep in dependencies:
            cmd.extend(["-d", dep])
        for out in outputs:
            cmd.extend(["-o", out])
        if params:
            import tempfile

            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            )
            import yaml

            yaml.safe_dump(params, tmp)
            tmp.close()
            cmd.extend(["-p", tmp.name])

        cmd.append(command)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get(self, version_id: str) -> DatasetVersion:
        v = self._versions.get(version_id)
        if v is None:
            raise ValueError(f"Dataset version '{version_id}' not found")
        return v