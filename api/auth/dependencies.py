"""FastAPI auth dependencies (issue #240)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.auth.config import is_auth_enabled
from api.auth.security import ALL_SCOPES, decode_token, hash_api_key
from api.database import get_sync_db
from api.models.orm import ApiKey, User

_bearer = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    subject: str
    auth_type: str  # jwt | api_key | disabled
    scopes: list[str]
    user_id: Optional[int] = None


def _resolve_api_key(token: str, db: Session) -> AuthContext:
    key_hash = hash_api_key(token)
    api_key = db.scalar(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
    )
    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="API key expired")
    return AuthContext(
        subject=api_key.name,
        auth_type="api_key",
        scopes=api_key.scopes or [],
        user_id=api_key.user_id,
    )


def _resolve_jwt(token: str, db: Session) -> AuthContext:
    try:
        payload = decode_token(token)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    if payload.get("type") != "jwt":
        raise HTTPException(status_code=401, detail="Invalid token type")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token subject")

    user = db.scalar(select(User).where(User.username == username))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return AuthContext(
        subject=username,
        auth_type="jwt",
        scopes=user.scopes or [],
        user_id=user.id,
    )


def get_current_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_sync_db),
) -> AuthContext:
    if not is_auth_enabled():
        return AuthContext(subject="anonymous", auth_type="disabled", scopes=list(ALL_SCOPES))

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    if token.startswith("ak_"):
        return _resolve_api_key(token, db)
    return _resolve_jwt(token, db)


def require_scopes(*required: str):
    """Dependency factory that enforces scope membership."""

    def _checker(auth: AuthContext = Depends(get_current_auth)) -> AuthContext:
        if not is_auth_enabled():
            return auth
        if "admin" in auth.scopes:
            return auth
        missing = set(required) - set(auth.scopes)
        if missing:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required scopes: {', '.join(sorted(missing))}",
            )
        return auth

    return _checker


def authenticate_token(token: str, db: Session) -> AuthContext:
    """Validate a raw bearer token (used by WebSocket query-param auth)."""
    if not is_auth_enabled():
        return AuthContext(subject="anonymous", auth_type="disabled", scopes=list(ALL_SCOPES))
    if token.startswith("ak_"):
        return _resolve_api_key(token, db)
    return _resolve_jwt(token, db)


def get_current_user(
    auth: AuthContext = Depends(get_current_auth),
    db: Session = Depends(get_sync_db),
) -> User:
    """Resolve the currently authenticated User database model."""
    if not is_auth_enabled():
        # Return a mock admin user if auth is disabled
        user = db.scalar(select(User).where(User.username == "admin"))
        if not user:
            user = User(username="admin", email="admin@astroml.dev", scopes=list(ALL_SCOPES), is_active=True)
        return user

    if auth.user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = db.scalar(select(User).where(User.id == auth.user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

