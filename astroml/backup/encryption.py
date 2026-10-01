"""Backup encryption at rest (issue #958).

Local and cloud-uploaded backup archives (``.sql.gz`` / ``.tar.gz``) are
written to disk with no confidentiality protection: a compromised backup
host, misconfigured S3/GCS bucket, or leaked local-disk snapshot exposes raw
database dumps (which can contain PII per issue #960) and model artifacts in
plaintext. This module adds opt-in, authenticated symmetric encryption for
backup archives using :mod:`cryptography`'s Fernet (AES-128-CBC with an
HMAC-SHA256 integrity tag), so a corrupted or tampered ciphertext fails
loudly instead of silently decrypting to garbage.

Encryption is off by default (``BackupConfig.encryption_key is None``) to
preserve existing behavior for callers who haven't configured a key;
:func:`astroml.backup.service.BackupService` and
:func:`astroml.backup.restore.RestoreService` both consult
:func:`is_encryption_enabled` before touching ciphertext.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.fernet import Fernet

    from .service import BackupConfig

#: Suffix appended to an encrypted backup file's name, after its existing
#: ``.sql.gz`` / ``.tar.gz`` extension (e.g. ``db_20260925.sql.gz.enc``).
ENCRYPTED_SUFFIX = ".enc"


class BackupEncryptionError(RuntimeError):
    """Raised when a backup cannot be encrypted or decrypted.

    Wraps the underlying :mod:`cryptography` failure (or a missing/invalid
    key) so callers can catch one exception type regardless of cause.
    """


def is_encryption_enabled(config: BackupConfig) -> bool:
    """Return True when ``config`` has a usable encryption key configured."""
    return bool(config.encryption_key)


def _get_fernet(key: str) -> Fernet:
    """Build a ``Fernet`` cipher from a base64 urlsafe key string.

    Args:
        key: A Fernet key, as produced by
            :func:`generate_encryption_key`.

    Returns:
        A configured ``Fernet`` instance.

    Raises:
        BackupEncryptionError: If ``cryptography`` is not installed or the
            key is malformed.
    """
    try:
        from cryptography.fernet import Fernet, InvalidToken  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised via lint/env, not unit tests
        raise BackupEncryptionError(
            "Backup encryption requires the 'cryptography' package. "
            "Install it (`pip install cryptography`) or unset "
            "BackupConfig.encryption_key to disable encryption."
        ) from e

    try:
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except (ValueError, TypeError) as e:
        raise BackupEncryptionError(f"Invalid backup encryption key: {e}") from e


def generate_encryption_key() -> str:
    """Generate a new, random Fernet key suitable for ``BackupConfig.encryption_key``.

    Returns:
        A base64 urlsafe-encoded 32-byte key, as a ``str``.

    Raises:
        BackupEncryptionError: If ``cryptography`` is not installed.
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError as e:  # pragma: no cover - exercised via lint/env, not unit tests
        raise BackupEncryptionError(
            "Backup encryption requires the 'cryptography' package. "
            "Install it (`pip install cryptography`)."
        ) from e
    return Fernet.generate_key().decode("utf-8")


def encrypt_file(source: Path, key: str) -> Path:
    """Encrypt ``source`` in place, replacing it with a ``.enc`` sibling.

    Args:
        source: Path to the plaintext backup archive.
        key: Fernet key from :func:`generate_encryption_key`.

    Returns:
        Path to the encrypted file (``source`` with :data:`ENCRYPTED_SUFFIX`
        appended). The original plaintext file is removed.

    Raises:
        BackupEncryptionError: If encryption fails for any reason (missing
            dependency, bad key, I/O error).
    """
    fernet = _get_fernet(key)
    encrypted_path = source.with_name(source.name + ENCRYPTED_SUFFIX)

    try:
        plaintext = source.read_bytes()
        ciphertext = fernet.encrypt(plaintext)
        encrypted_path.write_bytes(ciphertext)
    except OSError as e:
        raise BackupEncryptionError(f"Failed to encrypt {source}: {e}") from e

    source.unlink()
    return encrypted_path


def decrypt_file(source: Path, key: str) -> Path:
    """Decrypt ``source`` in place, replacing it with the plaintext original.

    Args:
        source: Path to an encrypted backup archive (ending in
            :data:`ENCRYPTED_SUFFIX`).
        key: Fernet key that was used to encrypt the file.

    Returns:
        Path to the decrypted plaintext file (``source`` with
        :data:`ENCRYPTED_SUFFIX` stripped). The encrypted file is removed.

    Raises:
        BackupEncryptionError: If decryption fails — wrong key, corrupted or
            tampered ciphertext, missing dependency, or I/O error.
    """
    from cryptography.fernet import InvalidToken

    fernet = _get_fernet(key)

    if source.suffix != ENCRYPTED_SUFFIX:
        raise BackupEncryptionError(
            f"Expected a '{ENCRYPTED_SUFFIX}' encrypted backup file, got: {source}"
        )
    plaintext_path = source.with_name(source.name[: -len(ENCRYPTED_SUFFIX)])

    try:
        ciphertext = source.read_bytes()
        plaintext = fernet.decrypt(ciphertext)
        plaintext_path.write_bytes(plaintext)
    except InvalidToken as e:
        raise BackupEncryptionError(
            f"Failed to decrypt {source}: invalid key or corrupted/tampered backup"
        ) from e
    except OSError as e:
        raise BackupEncryptionError(f"Failed to decrypt {source}: {e}") from e

    source.unlink()
    return plaintext_path


__all__ = [
    "ENCRYPTED_SUFFIX",
    "BackupEncryptionError",
    "is_encryption_enabled",
    "generate_encryption_key",
    "encrypt_file",
    "decrypt_file",
]
