"""Encryption at rest for backup files (issue #965).

Database dumps and model artifact archives can contain PII, credentials
embedded in seed data, and proprietary model weights. Before this module,
`BackupService` wrote them to disk (and to S3/GCS) gzip-compressed only,
which is not encryption: anyone with filesystem or bucket read access could
read a backup's contents directly.

Backups use `cryptography`'s `Fernet` (AES-128-CBC with an HMAC-SHA256
authentication tag) rather than the `python-jose` JWE helper already used
for API keys in `astroml/llm/secrets.py`: JWE's compact serialization is
built for small tokens, not multi-gigabyte database dumps, and `python-jose`
is not declared in any requirements file in this repo (a pre-existing gap,
left as-is here since fixing it is outside this issue's scope). Fernet
encrypts/decrypts a file's bytes directly with no size-related overhead
beyond a small fixed header per token.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

ENCRYPTED_SUFFIX = ".enc"


class BackupEncryptionError(RuntimeError):
    """Raised when a backup cannot be encrypted or decrypted."""


def get_backup_encryption_key() -> bytes:
    """Derive the Fernet key used for backup encryption.

    Reads `BACKUP_ENCRYPTION_KEY` (or falls back to `SECRET_KEY`, matching
    the fallback chain `astroml/llm/secrets.py` already uses for API key
    encryption). The raw secret is hashed to 32 bytes and base64url-encoded,
    since Fernet requires a 32-byte urlsafe-base64 key rather than an
    arbitrary-length passphrase.
    """
    secret = (
        os.getenv("BACKUP_ENCRYPTION_KEY")
        or os.getenv("SECRET_KEY")
        or "change-me-in-production-default-secret-key"
    )
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_backup_file(source_path: Path, *, delete_source: bool = True) -> Path:
    """Encrypt `source_path` in place, returning the path to the `.enc` file.

    The plaintext (here, already gzip- or tar.gz-compressed) file is read
    fully into memory, encrypted, and written to `source_path` with
    `ENCRYPTED_SUFFIX` appended. Reading the whole file is acceptable for
    backup archive sizes this service targets; a true streaming AEAD would
    be needed before this scales to multi-gigabyte dumps without matching
    memory headroom, which is out of scope for this fix.

    Args:
        source_path: Path to the plaintext backup file.
        delete_source: Remove the plaintext file after a successful
            encrypted write, so the unencrypted bytes never persist on
            disk. Defaults to True; tests that need to inspect both forms
            can pass False.

    Returns:
        Path to the encrypted file (`source_path` with `.enc` appended).
    """
    key = get_backup_encryption_key()
    fernet = Fernet(key)

    plaintext = source_path.read_bytes()
    token = fernet.encrypt(plaintext)

    encrypted_path = source_path.with_name(source_path.name + ENCRYPTED_SUFFIX)
    encrypted_path.write_bytes(token)

    if delete_source:
        source_path.unlink()

    logger.info(f"Encrypted backup file: {encrypted_path.name}")
    return encrypted_path


def decrypt_backup_file(encrypted_path: Path, *, delete_source: bool = False) -> Path:
    """Decrypt `encrypted_path`, returning the path to the plaintext file.

    The plaintext is written alongside the encrypted file with
    `ENCRYPTED_SUFFIX` stripped from the name, so callers that expect a
    `.sql.gz` or `.tar.gz` path (e.g. to pass to `gzip.open`/`tarfile.open`)
    get exactly that back.

    Args:
        encrypted_path: Path to the `.enc` backup file.
        delete_source: Remove the encrypted file after a successful
            decrypt. Defaults to False, since restore call sites generally
            want to keep the encrypted backup intact on disk/in cloud
            storage and only produce a transient plaintext copy.

    Returns:
        Path to the decrypted plaintext file.

    Raises:
        BackupEncryptionError: If the file cannot be decrypted, e.g. the
            encryption key is wrong or the file was corrupted or tampered
            with (Fernet's HMAC tag fails to verify).
    """
    key = get_backup_encryption_key()
    fernet = Fernet(key)

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
