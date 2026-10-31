"""End-to-end backup -> restore regression tests (issue #889).

The unit tests in ``test_service_encryption.py`` verify that a backup is
encrypted and that a single restore call decrypts before handing bytes to
``psql``. These tests close the loop: they take a backup all the way through
:class:`~astroml.backup.restore.RestoreService` and assert the recovered
payload equals what went in, for both backup halves and for a full backup.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from astroml.backup.encryption import encrypt_backup_file
from astroml.backup.restore import RestoreService
from astroml.backup.service import BackupConfig, BackupService, BackupType, StorageBackend


@pytest.fixture(autouse=True)
def _fixed_encryption_key(monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "restore-e2e-fixed-key")


@pytest.fixture
def config(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return BackupConfig(
        database_url="postgresql://dbuser:dbpass@localhost:5432/astroml",
        database_name="astroml",
        storage_backend=StorageBackend.LOCAL,
        local_backup_dir=str(tmp_path / "backups"),
        verify_after_backup=True,
        model_artifacts_dir=str(artifacts),
    )


def _fake_pg_dump_result(sql_text: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["pg_dump"], returncode=0, stdout=sql_text, stderr="")


class _FakeProcess:
    """Stand-in for the ``psql`` restore subprocess."""

    returncode = 0

    def __init__(self, captured: dict[str, str]) -> None:
        self._captured = captured

    def communicate(self, input: str | None = None):  # noqa: A002 - mirrors Popen
        self._captured["sql"] = input or ""
        return ("", "")


def _create_database_backup(config, sql_text):
    with patch(
        "astroml.backup.service.subprocess.run",
        return_value=_fake_pg_dump_result(sql_text),
    ):
        return BackupService(config).create_database_backup(description="e2e")


class TestDatabaseRestoreE2E:
    def test_database_backup_restores_the_original_sql(self, config):
        secret_sql = "CREATE TABLE ledgers (id int);\nINSERT INTO ledgers VALUES (1);"
        metadata = _create_database_backup(config, secret_sql)

        captured: dict[str, str] = {}
        restore = RestoreService(config)
        with (
            patch("astroml.backup.restore.subprocess.run"),
            patch(
                "astroml.backup.restore.subprocess.Popen",
                side_effect=lambda *a, **k: _FakeProcess(captured),
            ),
        ):
            assert restore.restore_database(metadata.backup_id) is True

        assert captured["sql"] == secret_sql
        # The transient plaintext copy must not survive the restore.
        assert not Path(metadata.storage_path[: -len(".enc")]).exists()

    def test_database_restore_returns_false_for_missing_backup(self, config):
        restore = RestoreService(config)
        assert restore.restore_database("db_does_not_exist") is False


class TestModelArtifactRestoreE2E:
    def _seed_artifacts(self, config):
        artifacts = Path(config.model_artifacts_dir)
        (artifacts / "model.pt").write_bytes(b"\x00\x01model-weights")
        nested = artifacts / "config"
        nested.mkdir()
        (nested / "train.yaml").write_text("seed: 42\n", encoding="utf-8")
        return artifacts

    def test_model_backup_restores_every_file(self, config):
        self._seed_artifacts(config)
        service = BackupService(config)
        metadata = service.create_model_backup()

        dest = Path(config.model_artifacts_dir) / "restored"
        assert (
            RestoreService(config).restore_model_artifacts(metadata.backup_id, target_dir=str(dest))
            is True
        )

        assert (dest / "model.pt").read_bytes() == b"\x00\x01model-weights"
        assert (dest / "config" / "train.yaml").read_text(encoding="utf-8") == "seed: 42\n"

    def test_model_restore_rejects_a_directory_outside_the_artifacts_root(self, config, tmp_path):
        self._seed_artifacts(config)
        metadata = BackupService(config).create_model_backup()

        # ``artifacts-evil`` shares the string prefix of ``artifacts`` but is
        # not inside it; a naive ``str.startswith`` check would accept it.
        evil = tmp_path / "artifacts-evil"
        evil.mkdir()
        assert (
            RestoreService(config).restore_model_artifacts(metadata.backup_id, target_dir=str(evil))
            is False
        )
        assert not any(evil.iterdir())

    def test_model_restore_rejects_an_archive_member_that_escapes(self, config, tmp_path):
        self._seed_artifacts(config)
        service = BackupService(config)
        metadata = service.create_model_backup()

        # Replace the (valid) encrypted archive with a malicious one that
        # tries to write a file one level above the restore target.
        source_dir = tmp_path / "payload-src"
        source_dir.mkdir()
        escape = source_dir / "escape.txt"
        escape.write_text("pwned", encoding="utf-8")

        payload = tmp_path / "malicious.tar.gz"
        with tarfile.open(payload, "w:gz") as tar:
            tar.add(escape, arcname="../escape.txt")

        config_data = Path(metadata.storage_path[: -len(".enc")])
        config_data.write_bytes(payload.read_bytes())
        encrypted = encrypt_backup_file(config_data)

        meta_file = Path(config.local_backup_dir) / "metadata" / f"{metadata.backup_id}.json"
        meta = json.loads(meta_file.read_text())
        meta["storage_path"] = str(encrypted)
        meta_file.write_text(json.dumps(meta))

        assert RestoreService(config).restore_model_artifacts(metadata.backup_id) is False
        # ``../escape.txt`` would land beside the artifacts dir, not inside it.
        assert not (tmp_path / "escape.txt").exists()


class TestFullRestoreE2E:
    def test_full_backup_round_trips_both_halves(self, config):
        artifacts = Path(config.model_artifacts_dir)
        (artifacts / "weights.bin").write_bytes(b"weights")

        sql_text = "CREATE TABLE t (id int);"
        with patch(
            "astroml.backup.service.subprocess.run",
            return_value=_fake_pg_dump_result(sql_text),
        ):
            service = BackupService(config)
            db_backup, model_backup = service.create_full_backup()

        # create_full_backup must pair the two halves under one shared timestamp.
        assert db_backup.backup_id.replace("db_", "", 1) == model_backup.backup_id.replace(
            "models_", "", 1
        )

        captured: dict[str, str] = {}
        restore = RestoreService(config)
        with (
            patch("astroml.backup.restore.subprocess.run"),
            patch(
                "astroml.backup.restore.subprocess.Popen",
                side_effect=lambda *a, **k: _FakeProcess(captured),
            ),
        ):
            assert restore.restore_full(db_backup.backup_id) is True

        assert captured["sql"] == sql_text
        assert (artifacts / "weights.bin").read_bytes() == b"weights"

    def test_restore_full_accepts_the_model_half_id_too(self, config):
        artifacts = Path(config.model_artifacts_dir)
        (artifacts / "weights.bin").write_bytes(b"weights")
        with patch(
            "astroml.backup.service.subprocess.run",
            return_value=_fake_pg_dump_result("SELECT 1;"),
        ):
            service = BackupService(config)
            _, model_backup = service.create_full_backup()

        captured: dict[str, str] = {}
        with (
            patch("astroml.backup.restore.subprocess.run"),
            patch(
                "astroml.backup.restore.subprocess.Popen",
                side_effect=lambda *a, **k: _FakeProcess(captured),
            ),
        ):
            assert RestoreService(config).restore_full(model_backup.backup_id) is True
        assert captured["sql"] == "SELECT 1;"


def test_metadata_lists_both_halves_as_verified(config):
    artifacts = Path(config.model_artifacts_dir)
    (artifacts / "w.bin").write_bytes(b"w")
    with patch(
        "astroml.backup.service.subprocess.run",
        return_value=_fake_pg_dump_result("SELECT 1;"),
    ):
        service = BackupService(config)
        service.create_full_backup()

    assert len(service.list_backups(BackupType.DATABASE)) == 1
    assert len(service.list_backups(BackupType.MODEL_ARTIFACTS)) == 1
    assert all(b.is_verified for b in service.list_backups())


def test_gzip_round_trip_is_readable_after_restore(config):
    sql_text = "SELECT 1;"
    metadata = _create_database_backup(config, sql_text)

    from astroml.backup.encryption import decrypt_backup_file

    decrypted = decrypt_backup_file(Path(metadata.storage_path))
    with gzip.open(decrypted, "rt", encoding="utf-8") as handle:
        assert handle.read() == sql_text
