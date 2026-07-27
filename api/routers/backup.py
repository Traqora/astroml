"""Backup and restore router for issue #304.

Provides endpoints for:
- Creating database backups
- Creating model artifact backups
- Listing backups
- Restoring from backups
- Applying retention policy
- Backup verification
"""
from __future__ import annotations

import os
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from astroml.backup import BackupService, RestoreService, BackupConfig, BackupVerifier
from astroml.backup.service import BackupType, StorageBackend

router = APIRouter(prefix="/backup", tags=["backup"])


# ─── Request/Response Schemas ─────────────────────────────────────────────

class BackupRequest(BaseModel):
    """Request schema for creating a backup."""

    backup_type: str = Field(..., description="Type of backup: database, models, full")
    description: Optional[str] = Field(None, description="Optional backup description")


class RestoreRequest(BaseModel):
    """Request schema for restoring from backup."""

    backup_id: str = Field(..., description="ID of the backup to restore")
    backup_type: str = Field(..., description="Type of backup: database, models, full")
    drop_existing: bool = Field(default=False, description="Whether to drop existing database")
    target_dir: Optional[str] = Field(None, description="Target directory for model restore")


class BackupResponse(BaseModel):
    """Response schema for backup operations."""

    backup_id: str
    backup_type: str
    created_at: str
    size_bytes: int
    checksum: str
    storage_path: str
    is_verified: bool
    description: Optional[str] = None


class BackupListResponse(BaseModel):
    """Response schema for listing backups."""

    backups: List[BackupResponse]
    total: int


class RestoreResponse(BaseModel):
    """Response schema for restore operations."""

    success: bool
    message: str
    backup_id: str


# ─── Dependency for Backup Service ────────────────────────────────────────

def get_backup_config() -> BackupConfig:
    """Get backup configuration from environment variables."""
    return BackupConfig(
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://astroml:astroml@localhost/astroml"
        ),
        database_name=os.environ.get("DB_NAME", "astroml"),
        storage_backend=StorageBackend(
            os.environ.get("STORAGE_BACKEND", "local")
        ),
        local_backup_dir=os.environ.get("BACKUP_DIR", "/tmp/backups"),
        s3_bucket=os.environ.get("S3_BUCKET"),
        gcs_bucket=os.environ.get("GCS_BUCKET"),
        retention_days=int(os.environ.get("BACKUP_RETENTION_DAYS", "30")),
        max_backups=int(os.environ.get("MAX_BACKUPS", "10")),
        model_artifacts_dir=os.environ.get("MODEL_ARTIFACTS_DIR", "/tmp/model_artifacts"),
    )


# ─── Backup Endpoints ─────────────────────────────────────────────────────

@router.post("/create", response_model=BackupResponse)
async def create_backup(
    request: BackupRequest,
    background_tasks: BackgroundTasks,
    config: BackupConfig = Depends(get_backup_config),
):
    """Create a new backup."""
    backup_service = BackupService(config)

    try:
        if request.backup_type == "database":
            metadata = backup_service.create_database_backup(request.description)
        elif request.backup_type == "models":
            metadata = backup_service.create_model_backup(request.description)
        elif request.backup_type == "full":
            # For full backup, return the database backup metadata
            metadatas = backup_service.create_full_backup(request.description)
            metadata = metadatas[0]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown backup type: {request.backup_type}")

        return BackupResponse(
            backup_id=metadata.backup_id,
            backup_type=metadata.backup_type.value,
            created_at=metadata.created_at.isoformat(),
            size_bytes=metadata.size_bytes,
            checksum=metadata.checksum,
            storage_path=metadata.storage_path,
            is_verified=metadata.is_verified,
            description=metadata.description,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Backup creation failed")


@router.get("/list", response_model=BackupListResponse)
async def list_backups(
    backup_type: Optional[str] = None,
    config: BackupConfig = Depends(get_backup_config),
):
    """List all available backups."""
    backup_service = BackupService(config)

    try:
        type_filter = None
        if backup_type:
            type_filter = BackupType(backup_type)

        metadatas = backup_service.list_backups(type_filter)

        return BackupListResponse(
            backups=[
                BackupResponse(
                    backup_id=m.backup_id,
                    backup_type=m.backup_type.value,
                    created_at=m.created_at.isoformat(),
                    size_bytes=m.size_bytes,
                    checksum=m.checksum,
                    storage_path=m.storage_path,
                    is_verified=m.is_verified,
                    description=m.description,
                )
                for m in metadatas
            ],
            total=len(metadatas),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list backups")


@router.delete("/{backup_id}")
async def delete_backup(
    backup_id: str,
    config: BackupConfig = Depends(get_backup_config),
):
    """Delete a backup."""
    backup_service = BackupService(config)

    try:
        success = backup_service.delete_backup(backup_id)
        if not success:
            raise HTTPException(status_code=404, detail="Backup not found")

        return {"message": f"Backup {backup_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete backup")


@router.post("/restore", response_model=RestoreResponse)
async def restore_backup(
    request: RestoreRequest,
    config: BackupConfig = Depends(get_backup_config),
):
    """Restore from a backup."""
    restore_service = RestoreService(config)

    try:
        success = False
        if request.backup_type == "database":
            success = restore_service.restore_database(
                request.backup_id,
                request.drop_existing
            )
        elif request.backup_type == "models":
            success = restore_service.restore_model_artifacts(
                request.backup_id,
                request.target_dir
            )
        elif request.backup_type == "full":
            success = restore_service.restore_full(
                request.backup_id,
                request.drop_existing
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown backup type: {request.backup_type}")

        if success:
            return RestoreResponse(
                success=True,
                message=f"Restore from {request.backup_id} completed successfully",
                backup_id=request.backup_id,
            )
        else:
            return RestoreResponse(
                success=False,
                message=f"Restore from {request.backup_id} failed",
                backup_id=request.backup_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Restore failed")


@router.post("/retention/apply")
async def apply_retention_policy(
    background_tasks: BackgroundTasks,
    config: BackupConfig = Depends(get_backup_config),
):
    """Apply retention policy and delete old backups."""
    backup_service = BackupService(config)

    try:
        deleted_count = backup_service.apply_retention_policy()
        return {
            "message": f"Retention policy applied",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to apply retention policy")


@router.post("/verify/{backup_id}")
async def verify_backup(
    backup_id: str,
    config: BackupConfig = Depends(get_backup_config),
):
    """Verify backup integrity."""
    backup_service = BackupService(config)
    verifier = BackupVerifier(config)

    try:
        # Find backup metadata
        metadatas = backup_service.list_backups()
        backup_metadata = next((m for m in metadatas if m.backup_id == backup_id), None)

        if not backup_metadata:
            raise HTTPException(status_code=404, detail="Backup not found")

        backup_file = backup_metadata.storage_path
        is_valid = verifier.verify_backup(
            backup_file,
            backup_metadata.checksum
        )

        return {
            "backup_id": backup_id,
            "is_valid": is_valid,
            "checksum": backup_metadata.checksum,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Verification failed")


@router.post("/download/{backup_id}")
async def download_from_cloud(
    backup_id: str,
    config: BackupConfig = Depends(get_backup_config),
):
    """Download backup from cloud storage."""
    restore_service = RestoreService(config)

    try:
        success = restore_service.download_from_cloud(backup_id)

        if success:
            return {"message": f"Backup {backup_id} downloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Download failed")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Download failed")
