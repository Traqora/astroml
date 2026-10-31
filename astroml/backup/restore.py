"""Restore service for database and model artifacts (issue #304)."""

from __future__ import annotations

import gzip
import logging
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from astroml.utils.exceptions import AstroMLError

from astroml.utils.exceptions import AstroMLError

from .encryption import decrypt_backup_file
from .service import BackupConfig, BackupType

logger = logging.getLogger(__name__)

_KNOWN_BACKUP_PREFIXES = ("models_", "db_", "full_")


def _backup_base_id(backup_id: str) -> str:
    """Strip a known type prefix from ``backup_id``, returning the shared part.

    ``db_20240101_000000`` and ``models_20240101_000000`` share the base
    ``20240101_000000``, which is what lets :meth:`RestoreService.restore_full`
    resolve both halves of a full backup from either ID.
    """
    for prefix in _KNOWN_BACKUP_PREFIXES:
        if backup_id.startswith(prefix):
            return backup_id[len(prefix) :]
    return backup_id


def _is_within(child: Path, parent: Path) -> bool:
    """Whether ``child`` is ``parent`` or lives inside it.

    Uses path-component comparison rather than a string prefix: with a string
    prefix, ``/tmp/artifacts-evil`` would be accepted as "inside"
    ``/tmp/artifacts``, which is exactly the directory-traversal hole an
    archive member or a caller-supplied target could exploit.
    """
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


class RestoreService:
    """Service for restoring from backups."""

    def __init__(self, config: BackupConfig) -> None:
        """Initialize restore service.

        Args:
            config: Backup configuration.
        """
        self.config = config
        self.backup_dir = Path(config.local_backup_dir)

    def restore_database(self, backup_id: str, drop_existing: bool = False) -> bool:
        """Restore database from a backup.

        Args:
            backup_id: ID of the backup to restore.
            drop_existing: Whether to drop existing database before restore.

        Returns:
            True if restore was successful.
        """
        # Find backup metadata
        metadata_file = self.backup_dir / "metadata" / f"{backup_id}.json"
        if not metadata_file.exists():
            logger.error(f"Backup metadata not found: {backup_id}")
            return False

        import json

        with open(metadata_file) as f:
            data = json.load(f)

        if data["backup_type"] != BackupType.DATABASE.value:
            logger.error(f"Backup {backup_id} is not a database backup")
            return False

        backup_file = Path(data["storage_path"])
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False

        logger.info(f"Restoring database from backup: {backup_id}")

        plaintext_file: Path | None = None
        if data.get("is_encrypted"):
            plaintext_file = decrypt_backup_file(backup_file)
            backup_file = plaintext_file

        try:
            # Extract database connection info
            db_url = self.config.database_url
            if "postgresql://" in db_url:
                parts = db_url.replace("postgresql://", "").split("/")
                conn_part = parts[0]
                db_name = parts[1] if len(parts) > 1 else self.config.database_name

                user_pass = conn_part.split("@")[0]
                host_port = conn_part.split("@")[1] if "@" in conn_part else "localhost:5432"

                user = user_pass.split(":")[0] if ":" in user_pass else user_pass
                password = user_pass.split(":")[1] if ":" in user_pass else ""
                host = host_port.split(":")[0] if ":" in host_port else host_port
                port = host_port.split(":")[1] if ":" in host_port else "5432"

                # Set PGPASSWORD environment variable
                env = os.environ.copy()
                if password:
                    env["PGPASSWORD"] = password

                # Drop existing database if requested
                if drop_existing:
                    logger.info(f"Dropping existing database: {db_name}")
                    drop_cmd = [
                        "psql",
                        "-h",
                        host,
                        "-p",
                        port,
                        "-U",
                        user,
                        "-d",
                        "postgres",
                        "-c",
                        f"DROP DATABASE IF EXISTS {db_name}",
                    ]
                    subprocess.run(drop_cmd, env=env, check=True)

                    # Create fresh database
                    create_cmd = [
                        "psql",
                        "-h",
                        host,
                        "-p",
                        port,
                        "-U",
                        user,
                        "-d",
                        "postgres",
                        "-c",
                        f"CREATE DATABASE {db_name}",
                    ]
                    subprocess.run(create_cmd, env=env, check=True)

                # Decompress and restore
                with gzip.open(backup_file, "rt", encoding="utf-8") as f:
                    sql_content = f.read()

                # Use psql to restore
                restore_cmd = [
                    "psql",
                    "-h",
                    host,
                    "-p",
                    port,
                    "-U",
                    user,
                    "-d",
                    db_name,
                ]

                process = subprocess.Popen(
                    restore_cmd,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                stdout, stderr = process.communicate(input=sql_content)

                if process.returncode != 0:
                    logger.error(f"Database restore failed: {stderr}")
                    return False

                logger.info(f"Database restored successfully from backup: {backup_id}")
                return True

            else:
                raise ValueError(f"Unsupported database URL format: {db_url}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Database restore command failed: {e}")
            return False
        except AstroMLError as e:
            logger.error(f"Database restore failed: {e}")
            return False
        finally:
            # The decrypted copy is transient: it must not outlive this
            # restore attempt regardless of outcome, or the encryption
            # this module exists to provide is undone by a leftover file.
            if plaintext_file is not None:
                plaintext_file.unlink(missing_ok=True)

    def restore_model_artifacts(self, backup_id: str, target_dir: str | None = None) -> bool:
        """Restore model artifacts from a backup.

        Args:
            backup_id: ID of the backup to restore.
            target_dir: Optional target directory (defaults to configured artifacts dir).

        Returns:
            True if restore was successful.
        """
        # Find backup metadata
        metadata_file = self.backup_dir / "metadata" / f"{backup_id}.json"
        if not metadata_file.exists():
            logger.error(f"Backup metadata not found: {backup_id}")
            return False

        import json

        with open(metadata_file) as f:
            data = json.load(f)

        if data["backup_type"] != BackupType.MODEL_ARTIFACTS.value:
            logger.error(f"Backup {backup_id} is not a model artifacts backup")
            return False

        backup_file = Path(data["storage_path"])
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_file}")
            return False

        base_allowed = Path(self.config.model_artifacts_dir)
        target_path = Path(target_dir or self.config.model_artifacts_dir)
        if not _is_within(target_path, base_allowed):
            logger.error(f"Invalid target directory: {target_dir}")
            return False
        target_path = target_path.resolve()
        target_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Restoring model artifacts from backup: {backup_id}")

        plaintext_file: Path | None = None
        if data.get("is_encrypted"):
            plaintext_file = decrypt_backup_file(backup_file)
            backup_file = plaintext_file

        try:
            # Extract tar.gz archive. Reject any member that would escape the
            # target directory (``../`` or an absolute path) before extracting
            # a single file, so a tampered archive cannot write outside it.
            with tarfile.open(backup_file, "r:gz") as tar:
                for member in tar.getmembers():
                    member_path = (target_path / member.name).resolve()
                    if not _is_within(member_path, target_path):
                        raise ValueError(f"Invalid archive member path: {member.name}")
                tar.extractall(path=target_path)

            logger.info(f"Model artifacts restored successfully from backup: {backup_id}")
            return True

        except (AstroMLError, ValueError, tarfile.TarError) as e:
            logger.error(f"Model artifacts restore failed: {e}")
            return False
        finally:
            if plaintext_file is not None:
                plaintext_file.unlink(missing_ok=True)

    def restore_full(self, backup_id: str, drop_existing_db: bool = False) -> bool:
        """Restore full backup (database + model artifacts).

        Args:
            backup_id: Any ID belonging to the full backup — the shared base
                ID, or the ``db_``/``models_``/``full_`` component ID. The
                counterpart ID is derived from the shared base, matching how
                :meth:`~astroml.backup.service.BackupService.create_full_backup`
                names its two halves.
            drop_existing_db: Whether to drop existing database before restore.

        Returns:
            True if all restores were successful.
        """
        core_id = _backup_base_id(backup_id)
        db_success = self.restore_database(f"db_{core_id}", drop_existing_db)
        model_success = self.restore_model_artifacts(f"models_{core_id}")

        return db_success and model_success

    def download_from_cloud(self, backup_id: str) -> bool:
        """Download backup from cloud storage.

        Args:
            backup_id: ID of the backup to download.

        Returns:
            True if download was successful.
        """
        # Find backup metadata
        metadata_file = self.backup_dir / "metadata" / f"{backup_id}.json"
        if not metadata_file.exists():
            logger.error(f"Backup metadata not found: {backup_id}")
            return False

        import json

        with open(metadata_file) as f:
            data = json.load(f)

        storage_backend = data["storage_backend"]
        backup_file = Path(data["storage_path"])

        if storage_backend == "s3":
            return self._download_from_s3(backup_id, backup_file)
        elif storage_backend == "gcs":
            return self._download_from_gcs(backup_id, backup_file)
        else:
            logger.info(f"Backup is already local: {backup_id}")
            return True

    def _download_from_s3(self, backup_id: str, local_path: Path) -> bool:
        """Download backup from S3."""
        try:
            import boto3
            from botocore.exceptions import ClientError

            s3_client = boto3.client("s3")
            bucket = self.config.s3_bucket

            if not bucket:
                raise ValueError("S3 bucket not configured")

            s3_client.download_file(
                bucket,
                f"backups/{local_path.name}",
                str(local_path),
            )

            logger.info(f"Downloaded {backup_id} from S3")
            return True

        except ImportError:
            logger.warning("boto3 not installed")
            return False
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            return False

    def _download_from_gcs(self, backup_id: str, local_path: Path) -> bool:
        """Download backup from Google Cloud Storage."""
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(self.config.gcs_bucket)

            if not self.config.gcs_bucket:
                raise ValueError("GCS bucket not configured")

            blob = bucket.blob(f"backups/{local_path.name}")
            blob.download_to_filename(str(local_path))

            logger.info(f"Downloaded {backup_id} from GCS")
            return True

        except ImportError:
            logger.warning("google-cloud-storage not installed")
            return False
        except AstroMLError as e:
            logger.error(f"GCS download failed: {e}")
            return False
