"""Restore service for database and model artifacts (issue #304)."""

from __future__ import annotations

import contextlib
import gzip
import logging
import os
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .encryption import ENCRYPTED_SUFFIX, BackupEncryptionError, is_encryption_enabled
from .service import BackupConfig, BackupType

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _plaintext_backup_file(
    config: BackupConfig, backup_file: Path, *, is_encrypted: bool
) -> Iterator[Path]:
    """Yield a plaintext path for ``backup_file``, decrypting to a temp copy if needed.

    Leaves the on-disk encrypted backup untouched (restores may run more than
    once against the same archive) and always cleans up the temporary
    plaintext copy, including when the caller raises.

    Args:
        config: Backup configuration; must have ``encryption_key`` set when
            ``is_encrypted`` is True.
        backup_file: Path to the (possibly encrypted) backup archive.
        is_encrypted: Whether ``backup_file`` is Fernet-encrypted.

    Yields:
        A path to a plaintext archive: ``backup_file`` itself when not
        encrypted, otherwise a temporary decrypted copy.

    Raises:
        BackupEncryptionError: If the backup is encrypted but no key is
            configured, or decryption fails (wrong key / tampered archive).
    """
    if not is_encrypted:
        yield backup_file
        return

    if not is_encryption_enabled(config) or not config.encryption_key:
        raise BackupEncryptionError(
            f"Backup {backup_file} is encrypted but no `encryption_key` is "
            "configured on BackupConfig; cannot restore it."
        )

    from cryptography.fernet import Fernet, InvalidToken

    try:
        fernet = Fernet(config.encryption_key.encode("utf-8"))
    except (ValueError, TypeError) as e:
        raise BackupEncryptionError(f"Invalid backup encryption key: {e}") from e

    plaintext_name = (
        backup_file.name[: -len(ENCRYPTED_SUFFIX)]
        if backup_file.name.endswith(ENCRYPTED_SUFFIX)
        else backup_file.name
    )
    plaintext_suffix = "".join(Path(plaintext_name).suffixes) or ".tmp"

    tmp = tempfile.NamedTemporaryFile(suffix=plaintext_suffix, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        ciphertext = backup_file.read_bytes()
        try:
            plaintext = fernet.decrypt(ciphertext)
        except InvalidToken as e:
            raise BackupEncryptionError(
                f"Failed to decrypt {backup_file}: invalid key or corrupted/tampered backup"
            ) from e
        tmp_path.write_bytes(plaintext)
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


class RestoreService:
    """Service for restoring from backups."""

    def __init__(self, config: BackupConfig):
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

        try:
            with _plaintext_backup_file(
                self.config, backup_file, is_encrypted=data.get("is_encrypted", False)
            ) as plaintext_file:
                return self._restore_database_from_plaintext(
                    plaintext_file, drop_existing, backup_id
                )
        except BackupEncryptionError as e:
            logger.error(f"Database restore failed: {e}")
            return False

    def _restore_database_from_plaintext(
        self, backup_file: Path, drop_existing: bool, backup_id: str
    ) -> bool:
        """Run the actual psql restore against a plaintext ``.sql.gz`` file."""
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
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False

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

        base_allowed = Path(self.config.model_artifacts_dir).resolve()
        target_path = (Path(target_dir or self.config.model_artifacts_dir)).resolve()
        if target_path != base_allowed and base_allowed not in target_path.parents:
            logger.error(f"Invalid target directory: {target_dir}")
            return False
        target_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Restoring model artifacts from backup: {backup_id}")

        try:
            with _plaintext_backup_file(
                self.config, backup_file, is_encrypted=data.get("is_encrypted", False)
            ) as plaintext_file:
                # Extract tar.gz archive, guarding against path traversal:
                # reject absolute member names and any member whose resolved
                # path would land outside target_path (e.g. via `../`).
                with tarfile.open(plaintext_file, "r:gz") as tar:
                    for member in tar.getmembers():
                        if os.path.isabs(member.name) or ".." in Path(member.name).parts:
                            raise ValueError(f"Invalid archive member path: {member.name}")
                        member_path = (target_path / member.name).resolve()
                        if member_path != target_path and target_path not in member_path.parents:
                            raise ValueError(f"Invalid archive member path: {member.name}")
                    # Defence in depth: the "data" filter (PEP 706) additionally
                    # rejects symlink/hardlink members and special files that
                    # could write outside target_path despite the checks above.
                    tar.extractall(path=target_path, filter="data")

            logger.info(f"Model artifacts restored successfully from backup: {backup_id}")
            return True

        except BackupEncryptionError as e:
            logger.error(f"Model artifacts restore failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Model artifacts restore failed: {e}")
            return False

    def restore_full(self, backup_id: str, drop_existing_db: bool = False) -> bool:
        """Restore full backup (database + model artifacts).

        Args:
            backup_id: Base ID of the backup (without type suffix).
            drop_existing_db: Whether to drop existing database before restore.

        Returns:
            True if all restores were successful.
        """
        db_backup_id = backup_id.replace("models_", "db_")
        model_backup_id = backup_id.replace("db_", "models_")

        db_success = self.restore_database(db_backup_id, drop_existing_db)
        model_success = self.restore_model_artifacts(model_backup_id)

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
        except Exception as e:
            logger.error(f"GCS download failed: {e}")
            return False
