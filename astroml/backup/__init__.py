"""Backup and restore system for issue #304.

Provides automated backup and restore for:
- Database backups using pg_dump
- Model artifact backups
- S3/GCS integration for storage
- Backup integrity verification
- One-click restore functionality
- Encryption at rest for backup archives (#958)
"""

from __future__ import annotations

from .encryption import (
    BackupEncryptionError,
    decrypt_file,
    encrypt_file,
    generate_encryption_key,
    is_encryption_enabled,
)
from .restore import RestoreService
from .service import BackupConfig, BackupService
from .verification import BackupVerifier

__all__ = [
    "BackupService",
    "BackupConfig",
    "RestoreService",
    "BackupVerifier",
    "BackupEncryptionError",
    "generate_encryption_key",
    "encrypt_file",
    "decrypt_file",
    "is_encryption_enabled",
]
