"""Tests for backup encryption at rest (issue #958)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from astroml.backup.encryption import (
    ENCRYPTED_SUFFIX,
    BackupEncryptionError,
    decrypt_file,
    encrypt_file,
    generate_encryption_key,
    is_encryption_enabled,
)
from astroml.backup.restore import RestoreService
from astroml.backup.service import BackupConfig, BackupService, BackupType, StorageBackend


class TestGenerateEncryptionKey:
    def test_returns_string(self):
        key = generate_encryption_key()
        assert isinstance(key, str)

    def test_returns_distinct_keys(self):
        assert generate_encryption_key() != generate_encryption_key()

    def test_returned_key_is_valid_fernet_key(self):
        # A round trip with the generated key should work without raising.
        key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_bytes(b"hello")
            encrypted = encrypt_file(path, key)
            decrypted = decrypt_file(encrypted, key)
            assert decrypted.read_bytes() == b"hello"


class TestIsEncryptionEnabled:
    def _config(self, **overrides) -> BackupConfig:
        defaults = dict(database_url="postgresql://u:p@localhost/db", database_name="db")
        defaults.update(overrides)
        return BackupConfig(**defaults)

    def test_false_when_key_is_none(self):
        assert is_encryption_enabled(self._config(encryption_key=None)) is False

    def test_false_when_key_is_empty_string(self):
        assert is_encryption_enabled(self._config(encryption_key="")) is False

    def test_true_when_key_is_set(self):
        assert is_encryption_enabled(self._config(encryption_key=generate_encryption_key())) is True


class TestEncryptFile:
    def test_creates_encrypted_sibling_with_expected_suffix(self):
        key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup.sql.gz"
            path.write_bytes(b"plaintext sql dump")
            encrypted = encrypt_file(path, key)
            assert encrypted.name == f"backup.sql.gz{ENCRYPTED_SUFFIX}"
            assert encrypted.exists()

    def test_removes_original_plaintext_file(self):
        key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup.sql.gz"
            path.write_bytes(b"plaintext sql dump")
            encrypt_file(path, key)
            assert not path.exists()

    def test_encrypted_content_does_not_contain_plaintext(self):
        key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup.sql.gz"
            # Synthetic marker bytes (not a real credential) standing in for
            # PII a database dump could contain, per issue #960; the
            # assertion below verifies Fernet ciphertext never leaks it.
            sensitive_marker = b"synthetic-pii-marker-7f3a2c1d-jane-doe"
            path.write_bytes(sensitive_marker)
            encrypted = encrypt_file(path, key)
            ciphertext = encrypted.read_bytes()
            assert sensitive_marker not in ciphertext

    def test_raises_backup_encryption_error_on_invalid_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup.sql.gz"
            path.write_bytes(b"data")
            with pytest.raises(BackupEncryptionError):
                encrypt_file(path, "not-a-valid-fernet-key")

    def test_raises_backup_encryption_error_when_cryptography_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup.sql.gz"
            path.write_bytes(b"data")
            with patch.dict("sys.modules", {"cryptography.fernet": None}):
                with pytest.raises(BackupEncryptionError, match="cryptography"):
                    encrypt_file(path, generate_encryption_key())


class TestDecryptFile:
    def test_round_trips_binary_content(self):
        key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifacts.tar.gz"
            payload = bytes(range(256)) * 10
            path.write_bytes(payload)
            encrypted = encrypt_file(path, key)
            decrypted = decrypt_file(encrypted, key)
            assert decrypted.read_bytes() == payload

    def test_removes_encrypted_file_after_decrypt(self):
        key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifacts.tar.gz"
            path.write_bytes(b"data")
            encrypted = encrypt_file(path, key)
            decrypt_file(encrypted, key)
            assert not encrypted.exists()

    def test_wrong_key_raises_backup_encryption_error(self):
        key = generate_encryption_key()
        wrong_key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup.sql.gz"
            path.write_bytes(b"data")
            encrypted = encrypt_file(path, key)
            with pytest.raises(BackupEncryptionError):
                decrypt_file(encrypted, wrong_key)

    def test_tampered_ciphertext_raises_backup_encryption_error(self):
        key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup.sql.gz"
            path.write_bytes(b"data")
            encrypted = encrypt_file(path, key)
            corrupted = bytearray(encrypted.read_bytes())
            corrupted[-1] ^= 0xFF
            encrypted.write_bytes(bytes(corrupted))
            with pytest.raises(BackupEncryptionError):
                decrypt_file(encrypted, key)

    def test_rejects_path_without_encrypted_suffix(self):
        key = generate_encryption_key()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "backup.sql.gz"
            path.write_bytes(b"data")
            with pytest.raises(BackupEncryptionError):
                decrypt_file(path, key)


class TestBackupServiceEncryptionIntegration:
    """End-to-end coverage of encryption wired into BackupService/RestoreService."""

    def _config(self, tmpdir: str, **overrides) -> BackupConfig:
        defaults = dict(
            database_url="postgresql://u:p@localhost/db",
            database_name="db",
            local_backup_dir=tmpdir,
            verify_after_backup=False,
        )
        defaults.update(overrides)
        return BackupConfig(**defaults)

    def test_model_backup_is_plaintext_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as artifacts:
            (Path(artifacts) / "model.bin").write_bytes(b"weights")
            config = self._config(tmpdir, model_artifacts_dir=artifacts)
            service = BackupService(config)
            metadata = service.create_model_backup()
            assert metadata.is_encrypted is False
            assert Path(metadata.storage_path).suffix != ENCRYPTED_SUFFIX

    def test_model_backup_is_encrypted_when_key_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as artifacts:
            (Path(artifacts) / "model.bin").write_bytes(b"weights")
            key = generate_encryption_key()
            config = self._config(tmpdir, model_artifacts_dir=artifacts, encryption_key=key)
            service = BackupService(config)
            metadata = service.create_model_backup()
            assert metadata.is_encrypted is True
            assert metadata.storage_path.endswith(ENCRYPTED_SUFFIX)
            assert Path(metadata.storage_path).exists()

    def test_encrypted_model_backup_metadata_round_trips_through_list_backups(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as artifacts:
            (Path(artifacts) / "model.bin").write_bytes(b"weights")
            key = generate_encryption_key()
            config = self._config(tmpdir, model_artifacts_dir=artifacts, encryption_key=key)
            service = BackupService(config)
            service.create_model_backup()

            listed = service.list_backups(backup_type=BackupType.MODEL_ARTIFACTS)
            assert len(listed) == 1
            assert listed[0].is_encrypted is True

    def test_restore_model_artifacts_decrypts_before_extraction_attempt(self):
        # Note: restore_model_artifacts()'s tar-member path-traversal guard
        # (astroml/backup/restore.py) resolves each member name via
        # `Path(member.name).resolve()`, which resolves relative to the
        # process CWD rather than to `target_dir`. That makes extraction fail
        # for effectively any caller whose CWD isn't the target directory —
        # a pre-existing bug, independent of encryption (reproduces
        # identically for plaintext backups) and out of scope for #958.
        #
        # What *is* in scope here: decryption must happen (and succeed)
        # before that extraction step runs, and must never leave a plaintext
        # temp file behind when extraction subsequently fails. We assert
        # that by patching tarfile.open to observe it was invoked with a
        # readable plaintext archive, without depending on the broken
        # extraction path succeeding end-to-end.
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as artifacts:
            (Path(artifacts) / "model.bin").write_bytes(b"weights-v1")
            key = generate_encryption_key()
            config = self._config(tmpdir, model_artifacts_dir=artifacts, encryption_key=key)
            service = BackupService(config)
            metadata = service.create_model_backup()

            import tarfile as tarfile_module

            original_open = tarfile_module.open
            opened_names = []
            member_names_seen = []

            def _spy_open(name=None, *args, **kwargs):
                opened_names.append(Path(name).name)
                tar = original_open(name, *args, **kwargs)
                # Record member names now, while the temp plaintext file
                # (deleted in a `finally` right after the `with` block in
                # restore.py) is still guaranteed to be open/readable.
                member_names_seen.extend(m.name for m in tar.getmembers())
                return tar

            restore_service = RestoreService(config)
            with patch.object(tarfile_module, "open", side_effect=_spy_open):
                restore_service.restore_model_artifacts(metadata.backup_id)

            assert len(opened_names) == 1
            assert not opened_names[0].endswith(ENCRYPTED_SUFFIX)
            # tarfile.open was called with a readable plaintext archive
            # (decryption succeeded and produced valid gzip/tar content)
            # before the pre-existing extraction-path bug rejected the member.
            assert member_names_seen == ["model.bin"]

            # The on-disk encrypted archive is left intact for repeat restores.
            assert Path(metadata.storage_path).exists()

    def test_restore_encrypted_backup_without_key_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as artifacts:
            (Path(artifacts) / "model.bin").write_bytes(b"weights")
            key = generate_encryption_key()
            config = self._config(tmpdir, model_artifacts_dir=artifacts, encryption_key=key)
            service = BackupService(config)
            metadata = service.create_model_backup()

            # No encryption_key configured on the restore side.
            restore_config = self._config(tmpdir, model_artifacts_dir=artifacts)
            restore_service = RestoreService(restore_config)
            success = restore_service.restore_model_artifacts(metadata.backup_id)
            assert success is False

    def test_restore_encrypted_backup_with_wrong_key_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as artifacts:
            (Path(artifacts) / "model.bin").write_bytes(b"weights")
            key = generate_encryption_key()
            wrong_key = generate_encryption_key()
            config = self._config(tmpdir, model_artifacts_dir=artifacts, encryption_key=key)
            service = BackupService(config)
            metadata = service.create_model_backup()

            restore_config = self._config(
                tmpdir, model_artifacts_dir=artifacts, encryption_key=wrong_key
            )
            restore_service = RestoreService(restore_config)
            success = restore_service.restore_model_artifacts(metadata.backup_id)
            assert success is False

    def test_no_leftover_plaintext_temp_file_after_restore(self):
        """Decrypting for restore must not leave a plaintext copy behind."""
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as artifacts:
            (Path(artifacts) / "model.bin").write_bytes(b"weights")
            key = generate_encryption_key()
            config = self._config(tmpdir, model_artifacts_dir=artifacts, encryption_key=key)
            service = BackupService(config)
            metadata = service.create_model_backup()

            restore_service = RestoreService(config)
            restore_service.restore_model_artifacts(metadata.backup_id)

            models_dir = Path(tmpdir) / "models"
            leftover_plaintext = [
                p for p in models_dir.iterdir() if not p.name.endswith(ENCRYPTED_SUFFIX)
            ]
            assert leftover_plaintext == []


class TestDatabaseBackupEncryptionIntegration:
    """Encryption coverage for create_database_backup, with pg_dump mocked out."""

    def _config(self, tmpdir: str, **overrides) -> BackupConfig:
        defaults = dict(
            database_url="postgresql://u:p@localhost:5432/astroml",
            database_name="astroml",
            local_backup_dir=tmpdir,
            verify_after_backup=False,
        )
        defaults.update(overrides)
        return BackupConfig(**defaults)

    def test_database_backup_is_encrypted_when_key_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            key = generate_encryption_key()
            config = self._config(tmpdir, encryption_key=key)
            service = BackupService(config)

            fake_result = type("R", (), {"stdout": "CREATE TABLE foo (id int);"})()
            with patch("subprocess.run", return_value=fake_result):
                metadata = service.create_database_backup()

            assert metadata.is_encrypted is True
            assert metadata.storage_path.endswith(ENCRYPTED_SUFFIX)
            ciphertext = Path(metadata.storage_path).read_bytes()
            assert b"CREATE TABLE" not in ciphertext

    def test_database_backup_checksum_is_over_plaintext_not_ciphertext(self):
        """Checksums (and verify_backup) must reflect the pg_dump output,
        not the encrypted bytes, since verification reads/parses SQL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key = generate_encryption_key()
            config = self._config(tmpdir, encryption_key=key)
            service = BackupService(config)

            fake_result = type("R", (), {"stdout": "CREATE TABLE foo (id int);"})()
            with patch("subprocess.run", return_value=fake_result):
                metadata = service.create_database_backup()

            decrypted = decrypt_file(
                Path(metadata.storage_path).with_name(Path(metadata.storage_path).name), key
            )
            import gzip
            import hashlib

            recomputed = hashlib.sha256(decrypted.read_bytes()).hexdigest()
            assert recomputed == metadata.checksum
            with gzip.open(decrypted, "rt") as f:
                assert "CREATE TABLE" in f.read()


class TestBackupMetadataIsEncryptedField:
    def test_defaults_to_false(self):
        from datetime import datetime

        from astroml.backup.service import BackupMetadata

        metadata = BackupMetadata(
            backup_id="db_1",
            backup_type=BackupType.DATABASE,
            created_at=datetime.utcnow(),
            size_bytes=10,
            checksum="abc",
            storage_path="/tmp/x",
            storage_backend=StorageBackend.LOCAL,
        )
        assert metadata.is_encrypted is False
        assert metadata.to_dict()["is_encrypted"] is False
