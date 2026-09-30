"""Authentication endpoints (issue #240, #534)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.auth.dependencies import AuthContext, get_current_auth, require_scopes
from api.auth.security import (
    ALL_SCOPES,
    api_key_expires_at,
    create_access_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    validate_scopes,
    verify_password,
)
from api.database import get_sync_db
from api.models.orm import ApiKey, AuditLog, User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int


class RefreshRequest(BaseModel):
    token: str


class ApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    scopes: list[str] = Field(default_factory=lambda: list(ALL_SCOPES))


class ApiKeyResponse(BaseModel):
    key: str
    name: str
    scopes: list[str]
    expires_at: datetime
    created_at: datetime


class RotateKeyRequest(BaseModel):
    name: str = Field(..., description="Name of the API key to rotate")


class RotateKeyResponse(BaseModel):
    new_key: str
    name: str
    scopes: list[str]
    expires_at: datetime
    created_at: datetime
    overlap_expires_at: datetime
    message: str


class RevokeKeyRequest(BaseModel):
    name: str = Field(..., description="Name of the API key to revoke")


class RevokeKeyResponse(BaseModel):
    name: str
    revoked: bool
    message: str


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_sync_db)):
    """Authenticate with username/password and return a JWT."""
    user = db.scalar(select(User).where(User.username == body.username))
    if (
        user is None
        or not user.is_active
        or not verify_password(body.password, user.hashed_password)
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(user.username, user.scopes or [])
    from api.auth.config import ACCESS_TOKEN_EXPIRE_HOURS  # noqa: PLC0415

    return TokenResponse(access_token=token, expires_in_hours=ACCESS_TOKEN_EXPIRE_HOURS)


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(body: RefreshRequest, db: Session = Depends(get_sync_db)):
    """Refresh a JWT before it expires."""
    try:
        payload = decode_token(body.token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token subject")

    user = db.scalar(select(User).where(User.username == username))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    token = create_access_token(username, user.scopes or [])
    from api.auth.config import ACCESS_TOKEN_EXPIRE_HOURS  # noqa: PLC0415

    return TokenResponse(access_token=token, expires_in_hours=ACCESS_TOKEN_EXPIRE_HOURS)


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    body: ApiKeyRequest,
    auth: AuthContext = Depends(require_scopes("admin")),
    db: Session = Depends(get_sync_db),
):
    """Generate a new API key for machine-to-machine access."""
    if auth.user_id is None:
        raise HTTPException(status_code=403, detail="API keys require a user account")

    scopes = validate_scopes(body.scopes)
    raw_key = generate_api_key()
    expires = api_key_expires_at()
    now = datetime.now(timezone.utc)

    entry = ApiKey(
        user_id=auth.user_id,
        key_hash=hash_api_key(raw_key),
        name=body.name,
        scopes=scopes,
        expires_at=expires,
        created_at=now,
    )
    db.add(entry)
    db.add(
        AuditLog(
            action="api_key.created",
            resource_type="api_key",
            resource_id=body.name,
            user_id=auth.user_id,
            username=auth.subject,
            auth_type=auth.auth_type,
            details={"scopes": scopes, "expires_at": expires.isoformat()},
        )
    )
    db.commit()
    db.refresh(entry)

    logger.info("api_key.created name=%s user_id=%s", body.name, auth.user_id)
    return ApiKeyResponse(
        key=raw_key,
        name=body.name,
        scopes=scopes,
        expires_at=expires,
        created_at=entry.created_at,
    )


@router.post("/rotate-key", response_model=RotateKeyResponse)
def rotate_api_key(
    body: RotateKeyRequest,
    auth: AuthContext = Depends(require_scopes("admin")),
    db: Session = Depends(get_sync_db),
):
    """Rotate an API key: generate a replacement and keep the old key valid
    for a 30-day overlap window so callers can migrate without downtime.

    POST /api/v1/auth/rotate-key
    {"name": "my-service-key"}
    """
    if auth.user_id is None:
        raise HTTPException(status_code=403, detail="API keys require a user account")

    api_key = db.scalar(
        select(ApiKey).where(
            ApiKey.name == body.name,
            ApiKey.user_id == auth.user_id,
            ApiKey.is_active.is_(True),
        )
    )
    if api_key is None:
        raise HTTPException(status_code=404, detail=f"Active API key '{body.name}' not found")

    from api.auth.config import API_KEY_ROTATION_OVERLAP_DAYS  # noqa: PLC0415

    now = datetime.now(timezone.utc)
    overlap_expires = now + timedelta(days=API_KEY_ROTATION_OVERLAP_DAYS)

    # Save the old key hash for the overlap window
    old_key_hash = api_key.key_hash

    # Issue a new key
    new_raw_key = generate_api_key()
    new_expires = api_key_expires_at()

    # Swap: new key becomes primary, old key stored in overlap slot
    api_key.key_hash = hash_api_key(new_raw_key)
    api_key.expires_at = new_expires
    api_key.created_at = now
    api_key.overlap_key_hash = old_key_hash
    api_key.overlap_expires_at = overlap_expires

    db.add(
        AuditLog(
            action="api_key.rotated",
            resource_type="api_key",
            resource_id=body.name,
            user_id=auth.user_id,
            username=auth.subject,
            auth_type=auth.auth_type,
            details={"overlap_expires_at": overlap_expires.isoformat()},
        )
    )
    db.commit()
    db.refresh(api_key)

    logger.info(
        "api_key.rotated name=%s user_id=%s overlap_expires=%s",
        body.name,
        auth.user_id,
        overlap_expires.isoformat(),
    )

    return RotateKeyResponse(
        new_key=new_raw_key,
        name=api_key.name,
        scopes=api_key.scopes or [],
        expires_at=new_expires,
        created_at=now,
        overlap_expires_at=overlap_expires,
        message=(
            f"Key rotated. Old key remains valid until {overlap_expires.date().isoformat()} "
            f"({API_KEY_ROTATION_OVERLAP_DAYS}-day overlap). Update your clients before that date."
        ),
    )


@router.post("/revoke-key", response_model=RevokeKeyResponse)
def revoke_api_key(
    body: RevokeKeyRequest,
    auth: AuthContext = Depends(require_scopes("admin")),
    db: Session = Depends(get_sync_db),
):
    """Immediately revoke an API key (sets is_active=False and clears overlap).

    POST /api/v1/auth/revoke-key
    {"name": "my-service-key"}
    """
    if auth.user_id is None:
        raise HTTPException(status_code=403, detail="API keys require a user account")

    api_key = db.scalar(
        select(ApiKey).where(
            ApiKey.name == body.name,
            ApiKey.user_id == auth.user_id,
        )
    )
    if api_key is None:
        raise HTTPException(status_code=404, detail=f"API key '{body.name}' not found")

    if not api_key.is_active:
        return RevokeKeyResponse(
            name=body.name,
            revoked=False,
            message=f"Key '{body.name}' was already revoked.",
        )

    api_key.is_active = False
    api_key.overlap_key_hash = None
    api_key.overlap_expires_at = None
    db.add(
        AuditLog(
            action="api_key.revoked",
            resource_type="api_key",
            resource_id=body.name,
            user_id=auth.user_id,
            username=auth.subject,
            auth_type=auth.auth_type,
            details={"revoked": True},
        )
    )
    db.commit()

    logger.info("api_key.revoked name=%s user_id=%s", body.name, auth.user_id)

    return RevokeKeyResponse(
        name=body.name,
        revoked=True,
        message=f"Key '{body.name}' has been revoked immediately. Both primary and overlap keys are now invalid.",
    )


def ensure_default_admin(db: Session) -> None:
    """Seed a default admin user when the table is empty."""
    from api.auth.config import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USERNAME  # noqa: PLC0415

    if db.scalar(select(User).limit(1)) is not None:
        return

    db.add(
        User(
            username=DEFAULT_ADMIN_USERNAME,
            hashed_password=hash_password(DEFAULT_ADMIN_PASSWORD),
            scopes=["admin", "read:transactions", "read:fraud", "write:loyalty"],
        )
    )
    db.commit()
