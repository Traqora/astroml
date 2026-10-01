"""Path-safety tests for RestoreService.restore_model_artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path

from astroml.backup.restore import RestoreService
from astroml.backup.service import BackupConfig, BackupService


def _config(backup_dir: str, artifacts: str) -> BackupConfig:
    return BackupConfig(
        database_url="postgresql://u:p@localhost/db",
        database_name="db",
        backup_dir=backup_dir,
        model_artifacts_dir=artifacts,
    )


def test_rejects_sibling_directory_sharing_name_prefix():
    # "<artifacts>_evil" starts with the string "<artifacts>" but is not inside it.
    with tempfile.TemporaryDirectory() as backups, tempfile.TemporaryDirectory() as root:
        artifacts = Path(root) / "models"
        artifacts.mkdir()
        (artifacts / "model.bin").write_bytes(b"w")
        evil = Path(root) / "models_evil"

        config = _config(backups, str(artifacts))
        metadata = BackupService(config).create_model_backup()

        assert RestoreService(config).restore_model_artifacts(
            metadata.backup_id, target_dir=str(evil)
        ) is False
        assert not evil.exists()


def test_rejects_parent_traversal_target_dir():
    with tempfile.TemporaryDirectory() as backups, tempfile.TemporaryDirectory() as root:
        artifacts = Path(root) / "models"
        artifacts.mkdir()
        (artifacts / "model.bin").write_bytes(b"w")

        config = _config(backups, str(artifacts))
        metadata = BackupService(config).create_model_backup()

        assert RestoreService(config).restore_model_artifacts(
            metadata.backup_id, target_dir=str(artifacts / ".." / "elsewhere")
        ) is False
