"""Regression tests for backup encryption at rest (issue #965)."""

from __future__ import annotations

import gzip
import os

import pytest

from astroml.backup.encryption import (
    BackupEncryptionError,
    decrypt_backup_file,
    encrypt_backup_file,
    get_backup_encryption_key,
)


@pytest.fixture(autouse=True)
def _fixed_encryption_key(monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "test-suite-fixed-backup-key")


def test_encrypt_backup_file_produces_non_plaintext_ciphertext(tmp_path):
    # This is the core regression: before issue #965, a backup written by
    # BackupService.create_database_backup was only gzip-compressed, so
    # anyone with filesystem access could read its contents directly.
    plaintext_path = tmp_path / "dump.sql.gz"
    with gzip.open(plaintext_path, "wt", encoding="utf-8") as f:
        f.write("CREATE TABLE users (id INT, email TEXT, ssn TEXT);\n")
        f.write("INSERT INTO users VALUES (1, 'alice@example.com', '123-45-6789');\n")

    encrypted_path = encrypt_backup_file(plaintext_path, delete_source=False)

    assert encrypted_path.name == "dump.sql.gz.enc"
    ciphertext = encrypted_path.read_bytes()
    assert b"alice@example.com" not in ciphertext
    assert b"123-45-6789" not in ciphertext
    assert b"CREATE TABLE" not in ciphertext


def test_encrypt_backup_file_deletes_plaintext_by_default(tmp_path):
    plaintext_path = tmp_path / "dump.sql.gz"
    plaintext_path.write_bytes(b"plaintext content")

    encrypted_path = encrypt_backup_file(plaintext_path)

    assert not plaintext_path.exists()
    assert encrypted_path.exists()


def test_decrypt_backup_file_recovers_the_original_bytes(tmp_path):
    plaintext_path = tmp_path / "archive.tar.gz"
    original = b"\x1f\x8b" + os.urandom(256)  # gzip magic + arbitrary bytes
    plaintext_path.write_bytes(original)

    encrypted_path = encrypt_backup_file(plaintext_path)
    decrypted_path = decrypt_backup_file(encrypted_path)

    assert decrypted_path.name == "archive.tar.gz"
    assert decrypted_path.read_bytes() == original


def test_decrypt_with_wrong_key_raises_instead_of_returning_garbage(tmp_path, monkeypatch):
    plaintext_path = tmp_path / "dump.sql.gz"
    plaintext_path.write_bytes(b"sensitive database contents")
    encrypted_path = encrypt_backup_file(plaintext_path)

    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "a-completely-different-key")

    with pytest.raises(BackupEncryptionError):
        decrypt_backup_file(encrypted_path)


def test_decrypt_a_corrupted_file_raises(tmp_path):
    plaintext_path = tmp_path / "dump.sql.gz"
    plaintext_path.write_bytes(b"sensitive database contents")
    encrypted_path = encrypt_backup_file(plaintext_path)

    # Flip a byte in the middle of the ciphertext to simulate corruption or
    # tampering; Fernet's HMAC tag must catch this rather than silently
    # decrypting to garbage.
    data = bytearray(encrypted_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    encrypted_path.write_bytes(bytes(data))

    with pytest.raises(BackupEncryptionError):
        decrypt_backup_file(encrypted_path)


def test_decrypt_rejects_a_path_without_the_encrypted_suffix(tmp_path):
    not_encrypted = tmp_path / "dump.sql.gz"
    not_encrypted.write_bytes(b"whatever")

    with pytest.raises(BackupEncryptionError):
        decrypt_backup_file(not_encrypted)


def test_get_backup_encryption_key_is_deterministic_for_the_same_secret(monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "same-secret")
    first = get_backup_encryption_key()
    second = get_backup_encryption_key()
    assert first == second


def test_get_backup_encryption_key_differs_across_secrets(monkeypatch):
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "secret-one")
    key_one = get_backup_encryption_key()
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "secret-two")
    key_two = get_backup_encryption_key()
    assert key_one != key_two


def test_get_backup_encryption_key_falls_back_to_secret_key(monkeypatch):
    monkeypatch.delenv("BACKUP_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("SECRET_KEY", "shared-app-secret")
    from_secret_key = get_backup_encryption_key()

    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", "shared-app-secret")
    from_backup_key = get_backup_encryption_key()

    assert from_secret_key == from_backup_key
