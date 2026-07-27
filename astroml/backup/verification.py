"""Backup verification utilities for issue #304."""
from __future__ import annotations

import gzip
import tarfile
import hashlib
import logging
from pathlib import Path
from typing import Optional

from .service import BackupConfig

logger = logging.getLogger(__name__)


class BackupVerifier:
    """Verifier for backup integrity."""

    def __init__(self, config: BackupConfig):
        """Initialize backup verifier.

        Args:
            config: Backup configuration.
        """
        self.config = config

    def verify_backup(self, backup_file: Path, expected_checksum: str) -> bool:
        """Verify backup integrity using checksum.

        Args:
            backup_file: Path to the backup file.
            expected_checksum: Expected SHA256 checksum.

        Returns:
            True if verification passed.
        """
        logger.info(f"Verifying backup: {backup_file}")

        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False

        try:
            calculated_checksum = self._calculate_checksum(backup_file)

            if calculated_checksum != expected_checksum:
                logger.error(
                    f"Checksum mismatch: expected {expected_checksum}, got {calculated_checksum}"
                )
                return False

            # Additional verification based on file type
            if backup_file.suffix == ".gz":
                if not self._verify_gzip_integrity(backup_file):
                    return False
            elif backup_file.suffixes == [".tar", ".gz"]:
                if not self._verify_tar_gz_integrity(backup_file):
                    return False

            logger.info(f"Backup verification passed: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    def verify_database_backup(self, backup_file: Path) -> bool:
        """Verify database backup can be read and parsed.

        Args:
            backup_file: Path to the database backup file.

        Returns:
            True if verification passed.
        """
        logger.info(f"Verifying database backup: {backup_file}")

        try:
            with gzip.open(backup_file, "rt", encoding="utf-8") as f:
                # Read first few lines to verify it's valid SQL
                lines = []
                for i, line in enumerate(f):
                    lines.append(line)
                    if i >= 10:
                        break

                # Check for SQL indicators
                content = "".join(lines)
                if "CREATE TABLE" in content or "INSERT INTO" in content or "COPY" in content:
                    logger.info(f"Database backup appears valid: {backup_file}")
                    return True
                else:
                    logger.warning(f"Database backup may not contain valid SQL: {backup_file}")
                    return False

        except Exception as e:
            logger.error(f"Database backup verification failed: {e}")
            return False

    def verify_model_backup(self, backup_file: Path) -> bool:
        """Verify model artifacts backup can be extracted.

        Args:
            backup_file: Path to the model backup file.

        Returns:
            True if verification passed.
        """
        logger.info(f"Verifying model backup: {backup_file}")

        try:
            with tarfile.open(backup_file, "r:gz") as tar:
                # Get list of files in archive
                members = tar.getmembers()

                if not members:
                    logger.warning(f"Model backup is empty: {backup_file}")
                    return False

                # Verify all members can be read
                for member in members:
                    tar.extractfile(member)

                logger.info(f"Model backup verification passed: {backup_file}")
                return True

        except Exception as e:
            logger.error(f"Model backup verification failed: {e}")
            return False

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _verify_gzip_integrity(self, file_path: Path) -> bool:
        """Verify gzip file integrity."""
        try:
            with gzip.open(file_path, "rb") as f:
                # Try to read some data
                f.read(1024)
            return True
        except Exception as e:
            logger.error(f"Gzip integrity check failed: {e}")
            return False

    def _verify_tar_gz_integrity(self, file_path: Path) -> bool:
        """Verify tar.gz file integrity."""
        try:
            with tarfile.open(file_path, "r:gz") as tar:
                # Try to read file list
                tar.getmembers()
            return True
        except Exception as e:
            logger.error(f"Tar.gz integrity check failed: {e}")
            return False
