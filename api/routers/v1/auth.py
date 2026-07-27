"""Authentication endpoints (issue #240)."""
from __future__ import annotations

from datetime import datetime
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
from api.models.orm import ApiKey, User

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


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


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_sync_db)):
    """Authenticate with username/password and return a JWT."""
    user = db.scalar(select(User).where(User.username == body.username))
    if user is None or not user.is_active or not verify_password(body.password, user.hashed_password):
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

    entry = ApiKey(
        user_id=auth.user_id,
        key_hash=hash_api_key(raw_key),
        name=body.name,
        scopes=scopes,
        expires_at=expires,
    )
    db.add(entry)
    db.commit()

    return ApiKeyResponse(key=raw_key, name=body.name, scopes=scopes, expires_at=expires)


def ensure_default_admin(db: Session) -> None:
    """Seed a default admin user when the table is empty."""
    from api.auth.config import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USERNAME  # noqa: PLC0415

    if db.scalar(select(User).limit(1)) is not None:
        return

    db.add(User(
        username=DEFAULT_ADMIN_USERNAME,
        hashed_password=hash_password(DEFAULT_ADMIN_PASSWORD),
        scopes=["admin", "read:transactions", "read:fraud", "write:loyalty"],
    ))
    db.commit()
