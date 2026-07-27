"""Backup service for database and model artifacts (issue #304)."""
from __future__ import annotations

import os
import subprocess
import gzip
import shutil
import hashlib
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""

    DATABASE = "database"
    MODEL_ARTIFACTS = "model_artifacts"
    FULL = "full"


class StorageBackend(Enum):
    """Storage backend options."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


@dataclass
class BackupConfig:
    """Configuration for backup service."""

    # Database configuration
    database_url: str
    database_name: str

    # Storage configuration
    storage_backend: StorageBackend = StorageBackend.LOCAL
    local_backup_dir: str = "/tmp/backups"
    s3_bucket: Optional[str] = None
    gcs_bucket: Optional[str] = None

    # Retention policy
    retention_days: int = 30
    max_backups: int = 10

    # Backup schedule
    schedule_enabled: bool = True
    schedule_interval_hours: int = 24

    # Verification
    verify_after_backup: bool = True

    # Model artifacts
    model_artifacts_dir: str = "/tmp/model_artifacts"


@dataclass
class BackupMetadata:
    """Metadata for a backup."""

    backup_id: str
    backup_type: BackupType
    created_at: datetime
    size_bytes: int
    checksum: str
    storage_path: str
    storage_backend: StorageBackend
    is_verified: bool = False
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "storage_path": self.storage_path,
            "storage_backend": self.storage_backend.value,
            "is_verified": self.is_verified,
            "description": self.description,
        }


class BackupService:
    """Service for creating and managing backups."""

    def __init__(self, config: BackupConfig):
        """Initialize backup service.

        Args:
            config: Backup configuration.
        """
        self.config = config
        self.backup_dir = Path(config.local_backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.backup_dir / "database").mkdir(exist_ok=True)
        (self.backup_dir / "models").mkdir(exist_ok=True)
        (self.backup_dir / "metadata").mkdir(exist_ok=True)

    def create_database_backup(self, description: Optional[str] = None) -> BackupMetadata:
        """Create a database backup using pg_dump.

        Args:
            description: Optional description for the backup.

        Returns:
            BackupMetadata with backup information.
        """
        backup_id = self._generate_backup_id("db")
        timestamp = datetime.utcnow()
        backup_file = self.backup_dir / "database" / f"{backup_id}.sql.gz"

        logger.info(f"Creating database backup: {backup_id}")

        try:
            # Extract database connection info
            db_url = self.config.database_url
            if "postgresql://" in db_url:
                # Parse postgresql://user:pass@host:port/db
                parts = db_url.replace("postgresql://", "").split("/")
                conn_part = parts[0]
                db_name = parts[1] if len(parts) > 1 else self.config.database_name

                user_pass = conn_part.split("@")[0]
                host_port = conn_part.split("@")[1] if "@" in conn_part else "localhost:5432"

                user = user_pass.split(":")[0] if ":" in user_pass else user_pass
                password = user_pass.split(":")[1] if ":" in user_pass else ""
                host = host_port.split(":")[0] if ":" in host_port else host_port
                port = host_port.split(":")[1] if ":" in host_port else "5432"

                # Set PGPASSWORD environment variable for pg_dump
                env = os.environ.copy()
                if password:
                    env["PGPASSWORD"] = password

                # Run pg_dump
                cmd = [
                    "pg_dump",
                    "-h", host,
                    "-p", port,
                    "-U", user,
                    "-d", db_name,
                    "-F", "p",  # Plain text format
                    "--no-owner",
                    "--no-acl",
                ]

                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Compress the output
                with gzip.open(backup_file, "wt", encoding="utf-8") as f:
                    f.write(result.stdout)

            else:
                raise ValueError(f"Unsupported database URL format: {db_url}")

            # Calculate checksum
            checksum = self._calculate_checksum(backup_file)
            size_bytes = backup_file.stat().st_size

            # Save metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=BackupType.DATABASE,
                created_at=timestamp,
                size_bytes=size_bytes,
                checksum=checksum,
                storage_path=str(backup_file),
                storage_backend=StorageBackend.LOCAL,
                is_verified=False,
                description=description,
            )

            self._save_metadata(metadata)

            # Upload to cloud storage if configured
            if self.config.storage_backend != StorageBackend.LOCAL:
                self._upload_to_storage(backup_file, backup_id)

            # Verify backup if enabled
            if self.config.verify_after_backup:
                from .verification import BackupVerifier
                verifier = BackupVerifier(self.config)
                metadata.is_verified = verifier.verify_backup(backup_file, checksum)
                self._save_metadata(metadata)

            logger.info(f"Database backup created successfully: {backup_id}")
            return metadata

        except subprocess.CalledProcessError as e:
            logger.error(f"pg_dump failed: {e.stderr}")
            raise RuntimeError(f"Database backup failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise

    def create_model_backup(self, description: Optional[str] = None) -> BackupMetadata:
        """Create a backup of model artifacts.

        Args:
            description: Optional description for the backup.

        Returns:
            BackupMetadata with backup information.
        """
        backup_id = self._generate_backup_id("models")
        timestamp = datetime.utcnow()
        artifacts_dir = Path(self.config.model_artifacts_dir)
        backup_file = self.backup_dir / "models" / f"{backup_id}.tar.gz"

        logger.info(f"Creating model artifacts backup: {backup_id}")

        if not artifacts_dir.exists():
            logger.warning(f"Model artifacts directory not found: {artifacts_dir}")
            # Create empty backup
            with tarfile.open(backup_file, "w:gz") as tar:
                pass
        else:
            import tarfile

            with tarfile.open(backup_file, "w:gz") as tar:
                for item in artifacts_dir.iterdir():
                    tar.add(item, arcname=item.name)

        # Calculate checksum
        checksum = self._calculate_checksum(backup_file)
        size_bytes = backup_file.stat().st_size

        # Save metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.MODEL_ARTIFACTS,
            created_at=timestamp,
            size_bytes=size_bytes,
            checksum=checksum,
            storage_path=str(backup_file),
            storage_backend=StorageBackend.LOCAL,
            is_verified=False,
            description=description,
        )

        self._save_metadata(metadata)

        # Upload to cloud storage if configured
        if self.config.storage_backend != StorageBackend.LOCAL:
            self._upload_to_storage(backup_file, backup_id)

        logger.info(f"Model artifacts backup created successfully: {backup_id}")
        return metadata

    def create_full_backup(self, description: Optional[str] = None) -> List[BackupMetadata]:
        """Create a full backup (database + model artifacts).

        Args:
            description: Optional description for the backup.

        Returns:
            List of BackupMetadata for each backup component.
        """
        logger.info("Creating full backup")
        db_backup = self.create_database_backup(description)
        model_backup = self.create_model_backup(description)
        return [db_backup, model_backup]

    def list_backups(self, backup_type: Optional[BackupType] = None) -> List[BackupMetadata]:
        """List all available backups.

        Args:
            backup_type: Optional filter by backup type.

        Returns:
            List of BackupMetadata.
        """
        metadata_dir = self.backup_dir / "metadata"
        backups = []

        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, "r") as f:
                data = json.load(f)
                metadata = BackupMetadata(
                    backup_id=data["backup_id"],
                    backup_type=BackupType(data["backup_type"]),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    size_bytes=data["size_bytes"],
                    checksum=data["checksum"],
                    storage_path=data["storage_path"],
                    storage_backend=StorageBackend(data["storage_backend"]),
                    is_verified=data.get("is_verified", False),
                    description=data.get("description"),
                )

                if backup_type is None or metadata.backup_type == backup_type:
                    backups.append(metadata)

        # Sort by creation date, newest first
        backups.sort(key=lambda x: x.created_at, reverse=True)
        return backups

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup.

        Args:
            backup_id: ID of the backup to delete.

        Returns:
            True if deleted successfully.
        """
        # Find metadata
        metadata_file = self.backup_dir / "metadata" / f"{backup_id}.json"
        if not metadata_file.exists():
            return False

        with open(metadata_file, "r") as f:
            data = json.load(f)
            storage_path = data["storage_path"]

        # Delete backup file
        backup_file = Path(storage_path)
        if backup_file.exists():
            backup_file.unlink()

        # Delete metadata
        metadata_file.unlink()

        logger.info(f"Backup deleted: {backup_id}")
        return True

    def apply_retention_policy(self) -> int:
        """Apply retention policy and delete old backups.

        Returns:
            Number of backups deleted.
        """
        backups = self.list_backups()
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
        deleted_count = 0

        for backup in backups:
            if backup.created_at < cutoff_date:
                if self.delete_backup(backup.backup_id):
                    deleted_count += 1

        # Also enforce max backups limit
        backups = self.list_backups()
        while len(backups) > self.config.max_backups:
            oldest = backups[-1]
            if self.delete_backup(oldest.backup_id):
                deleted_count += 1
            backups.pop()

        logger.info(f"Retention policy applied: {deleted_count} backups deleted")
        return deleted_count

    def _generate_backup_id(self, prefix: str) -> str:
        """Generate a unique backup ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _save_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to file."""
        metadata_file = self.backup_dir / "metadata" / f"{metadata.backup_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _upload_to_storage(self, file_path: Path, backup_id: str) -> None:
        """Upload backup to cloud storage (S3/GCS)."""
        if self.config.storage_backend == StorageBackend.S3:
            self._upload_to_s3(file_path, backup_id)
        elif self.config.storage_backend == StorageBackend.GCS:
            self._upload_to_gcs(file_path, backup_id)

    def _upload_to_s3(self, file_path: Path, backup_id: str) -> None:
        """Upload backup to S3."""
        try:
            import boto3
            from botocore.exceptions import ClientError

            s3_client = boto3.client("s3")
            bucket = self.config.s3_bucket

            if not bucket:
                raise ValueError("S3 bucket not configured")

            s3_client.upload_file(
                str(file_path),
                bucket,
                f"backups/{file_path.name}",
            )

            logger.info(f"Uploaded {backup_id} to S3 bucket {bucket}")

        except ImportError:
            logger.warning("boto3 not installed, skipping S3 upload")
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")

    def _upload_to_gcs(self, file_path: Path, backup_id: str) -> None:
        """Upload backup to Google Cloud Storage."""
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(self.config.gcs_bucket)

            if not self.config.gcs_bucket:
                raise ValueError("GCS bucket not configured")

            blob = bucket.blob(f"backups/{file_path.name}")
            blob.upload_from_filename(str(file_path))

            logger.info(f"Uploaded {backup_id} to GCS bucket {self.config.gcs_bucket}")

        except ImportError:
            logger.warning("google-cloud-storage not installed, skipping GCS upload")
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
