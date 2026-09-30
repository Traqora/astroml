"""Regression tests: BackupService writes encrypted backups end to end (issue #965)."""

from __future__ import annotations

import gzip
import subprocess
from unittest.mock import patch

import pytest

from astroml.backup.encryption import decrypt_backup_file
from astroml.backup.restore import RestoreService
from astroml.backup.service import BackupConfig, BackupService, BackupType, StorageBackend


@pytest.fixture(autouse=True)
def _fixed_encryption_key(monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "test-suite-fixed-backup-key")


@pytest.fixture
def config(tmp_path):
    return BackupConfig(
        database_url="postgresql://dbuser:dbpass@localhost:5432/astroml",
        database_name="astroml",
        storage_backend=StorageBackend.LOCAL,
        local_backup_dir=str(tmp_path / "backups"),
        verify_after_backup=True,
        model_artifacts_dir=str(tmp_path / "artifacts"),
    )


def _fake_pg_dump_result(sql_text: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["pg_dump"], returncode=0, stdout=sql_text, stderr="")


def test_create_database_backup_writes_an_encrypted_file_not_plaintext_sql(config):
    secret_sql = "INSERT INTO users VALUES (1, 'alice@example.com', 'super-secret-password-hash');"

    with patch(
        "astroml.backup.service.subprocess.run", return_value=_fake_pg_dump_result(secret_sql)
    ):
        service = BackupService(config)
        metadata = service.create_database_backup(description="test backup")

    assert metadata.is_encrypted is True
    assert metadata.storage_path.endswith(".enc")

    raw_bytes = open(metadata.storage_path, "rb").read()
    assert b"alice@example.com" not in raw_bytes
    assert b"super-secret-password-hash" not in raw_bytes

    # And the previous gzip-only plaintext file must not be left behind
    # alongside the encrypted one.
    plaintext_sibling = metadata.storage_path[: -len(".enc")]
    import os

    assert not os.path.exists(plaintext_sibling)


def test_create_database_backup_metadata_round_trips_is_encrypted(config):
    with patch(
        "astroml.backup.service.subprocess.run", return_value=_fake_pg_dump_result("SELECT 1;")
    ):
        service = BackupService(config)
        created = service.create_database_backup()

    backups = service.list_backups(backup_type=BackupType.DATABASE)
    assert len(backups) == 1
    assert backups[0].backup_id == created.backup_id
    assert backups[0].is_encrypted is True


def test_restore_database_decrypts_before_restoring(config):
    secret_sql = "INSERT INTO users VALUES (1, 'bob@example.com');"

    with patch(
        "astroml.backup.service.subprocess.run", return_value=_fake_pg_dump_result(secret_sql)
    ):
        service = BackupService(config)
        metadata = service.create_database_backup()

    restore_service = RestoreService(config)

    captured = {}

    class _FakeProcess:
        returncode = 0

        def communicate(self, input=None):
            captured["sql_sent_to_psql"] = input
            return ("", "")

    with (
        patch("astroml.backup.restore.subprocess.run"),
        patch("astroml.backup.restore.subprocess.Popen", return_value=_FakeProcess()),
    ):
        result = restore_service.restore_database(metadata.backup_id)

    assert result is True
    # The restore path must have decrypted the backup and fed the ORIGINAL
    # SQL to psql, proving decrypt_backup_file was actually exercised
    # rather than restore silently operating on ciphertext.
    assert captured["sql_sent_to_psql"] == secret_sql

    # And the transient plaintext copy must not survive the restore.
    import os

    plaintext_sibling = metadata.storage_path[: -len(".enc")]
    assert not os.path.exists(plaintext_sibling)


def test_create_model_backup_writes_an_encrypted_archive(config, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    secret_file = artifacts_dir / "model_config.json"
    secret_file.write_text('{"api_key": "sk-supersecretmodelkey"}')

    service = BackupService(config)
    metadata = service.create_model_backup()

    assert metadata.is_encrypted is True
    assert metadata.storage_path.endswith(".enc")

    raw_bytes = open(metadata.storage_path, "rb").read()
    assert b"sk-supersecretmodelkey" not in raw_bytes


def test_encrypted_backup_decrypts_back_to_a_valid_gzip_stream(config):
    with patch(
        "astroml.backup.service.subprocess.run", return_value=_fake_pg_dump_result("SELECT 1;")
    ):
        service = BackupService(config)
        metadata = service.create_database_backup()

    from pathlib import Path

    decrypted_path = decrypt_backup_file(Path(metadata.storage_path))
    with gzip.open(decrypted_path, "rt", encoding="utf-8") as f:
        content = f.read()
    assert "SELECT 1;" in content
