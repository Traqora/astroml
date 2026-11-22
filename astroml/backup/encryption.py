"""Encryption at rest for backup files (issues #958 and #965).

Database dumps and model artifact archives can contain PII, credentials
embedded in seed data, and proprietary model weights. A backup that is only
gzip-compressed is readable by anyone with filesystem or bucket access.

Both key sources are supported so existing deployments keep working:

* :func:`get_backup_encryption_key` derives a Fernet key from
  ``BACKUP_ENCRYPTION_KEY`` / ``SECRET_KEY`` (issue #965).
* :func:`generate_encryption_key` produces an explicit key for
  ``BackupConfig.encryption_key`` (issue #958), with opt-in helpers
  :func:`is_encryption_enabled`, :func:`encrypt_file` and
  :func:`decrypt_file`.

Fernet (AES-128-CBC with an HMAC-SHA256 tag) encrypts file bytes directly,
which suits multi-megabyte archives better than compact JWE tokens.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.fernet import Fernet

    from .service import BackupConfig

logger = logging.getLogger(__name__)

#: Suffix appended to an encrypted backup file's name, after its existing
#: ``.sql.gz`` / ``.tar.gz`` extension (e.g. ``db_20260925.sql.gz.enc``).
ENCRYPTED_SUFFIX = ".enc"


class BackupEncryptionError(RuntimeError):
    """Raised when a backup cannot be encrypted or decrypted.

    Wraps missing/invalid keys, missing ``cryptography``, I/O failures and
    Fernet integrity failures so callers can catch one exception type.
    """


def _load_fernet():
    """Import and return ``(Fernet, InvalidToken)`` with a friendly error."""
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ImportError as e:  # pragma: no cover - depends on environment
        raise BackupEncryptionError(
            "Backup encryption requires the 'cryptography' package. "
            "Install it (`pip install cryptography`) or disable encryption."
        ) from e
    return Fernet, InvalidToken


def get_backup_encryption_key() -> bytes:
    """Derive the Fernet key from ``BACKUP_ENCRYPTION_KEY`` / ``SECRET_KEY``.

    Falling back to ``SECRET_KEY`` matches the chain used for API key
    encryption in ``astroml.llm.secrets``. The raw secret is hashed to 32
    bytes and base64url-encoded because Fernet requires a 32-byte
    urlsafe-base64 key rather than an arbitrary-length passphrase.
    """
    secret = (
        os.getenv("BACKUP_ENCRYPTION_KEY")
        or os.getenv("SECRET_KEY")
        or "change-me-in-production-default-secret-key"
    )
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def is_env_encryption_enabled() -> bool:
    """True when an env-derived backup key is explicitly configured."""
    return bool(os.getenv("BACKUP_ENCRYPTION_KEY") or os.getenv("SECRET_KEY"))


def encrypt_backup_file(source_path: Path, *, delete_source: bool = True) -> Path:
    """Encrypt ``source_path`` with the env-derived key, returning the ``.enc`` path.

    Args:
        source_path: Path to the plaintext backup file.
        delete_source: Remove the plaintext after a successful encrypted
            write so unencrypted bytes never persist on disk.

    Returns:
        Path to the encrypted file (``source_path`` with ``.enc`` appended).
    """
    Fernet, _ = _load_fernet()
    fernet = Fernet(get_backup_encryption_key())

    plaintext = source_path.read_bytes()
    encrypted_path = source_path.with_name(source_path.name + ENCRYPTED_SUFFIX)
    encrypted_path.write_bytes(fernet.encrypt(plaintext))

    if delete_source:
        source_path.unlink()

    logger.info(f"Encrypted backup file: {encrypted_path.name}")
    return encrypted_path


def decrypt_backup_file(encrypted_path: Path, *, delete_source: bool = False) -> Path:
    """Decrypt ``encrypted_path`` with the env-derived key.

    The plaintext is written alongside the encrypted file with
    ``.enc`` stripped, so callers expecting a ``.sql.gz`` / ``.tar.gz``
    path get exactly that back.

    Raises:
        BackupEncryptionError: If the key is wrong or the file was
            corrupted or tampered with (Fernet's HMAC check fails).
    """
    Fernet, InvalidToken = _load_fernet()
    fernet = Fernet(get_backup_encryption_key())

    token = encrypted_path.read_bytes()
    try:
        plaintext = fernet.decrypt(token)
    except InvalidToken as exc:
        raise BackupEncryptionError(
            f"Failed to decrypt backup {encrypted_path.name}: wrong key or corrupted/tampered file"
        ) from exc

    if not encrypted_path.name.endswith(ENCRYPTED_SUFFIX):
        raise BackupEncryptionError(
            f"Expected an encrypted backup path ending in {ENCRYPTED_SUFFIX!r}, got {encrypted_path.name!r}"
        )
    plaintext_path = encrypted_path.with_name(encrypted_path.name[: -len(ENCRYPTED_SUFFIX)])
    plaintext_path.write_bytes(plaintext)

    if delete_source:
        encrypted_path.unlink()

    return plaintext_path


def is_encryption_enabled(config: BackupConfig) -> bool:
    """Return True when ``config`` has a usable explicit encryption key."""
    return bool(config.encryption_key)


def _get_fernet(key: str) -> Fernet:
    """Build a ``Fernet`` cipher from a base64 urlsafe key string."""
    Fernet, _ = _load_fernet()
    try:
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except (ValueError, TypeError) as e:
        raise BackupEncryptionError(f"Invalid backup encryption key: {e}") from e


def generate_encryption_key() -> str:
    """Generate a random Fernet key suitable for ``BackupConfig.encryption_key``."""
    Fernet, _ = _load_fernet()
    return Fernet.generate_key().decode("utf-8")


def encrypt_file(source: Path, key: str) -> Path:
    """Encrypt ``source`` with the explicit ``key``, replacing it with a ``.enc`` sibling."""
    fernet = _get_fernet(key)
    encrypted_path = source.with_name(source.name + ENCRYPTED_SUFFIX)

    try:
        encrypted_path.write_bytes(fernet.encrypt(source.read_bytes()))
    except OSError as e:
        raise BackupEncryptionError(f"Failed to encrypt {source}: {e}") from e

    source.unlink()
    return encrypted_path


def decrypt_file(source: Path, key: str) -> Path:
    """Decrypt ``source`` with the explicit ``key`` and remove the ciphertext.

    Raises:
        BackupEncryptionError: If the key is wrong, the ciphertext is
            corrupted/tampered, ``cryptography`` is missing, or I/O fails.
    """
    _, InvalidToken = _load_fernet()
    fernet = _get_fernet(key)

    if source.suffix != ENCRYPTED_SUFFIX:
        raise BackupEncryptionError(
            f"Expected a '{ENCRYPTED_SUFFIX}' encrypted backup file, got: {source}"
        )
    plaintext_path = source.with_name(source.name[: -len(ENCRYPTED_SUFFIX)])

    try:
        plaintext = fernet.decrypt(source.read_bytes())
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
    "get_backup_encryption_key",
    "is_env_encryption_enabled",
    "encrypt_backup_file",
    "decrypt_backup_file",
    "is_encryption_enabled",
    "generate_encryption_key",
    "encrypt_file",
    "decrypt_file",
]
