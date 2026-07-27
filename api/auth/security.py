"""JWT and password utilities (issue #240)."""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from api.auth.config import (
    ACCESS_TOKEN_EXPIRE_HOURS,
    ALGORITHM,
    API_KEY_EXPIRE_DAYS,
    SECRET_KEY,
)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

ALL_SCOPES = frozenset({
    "read:transactions",
    "read:fraud",
    "write:loyalty",
    "admin",
})


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(
    subject: str,
    scopes: list[str],
    expires_delta: Optional[timedelta] = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    )
    payload = {"sub": subject, "scopes": scopes, "exp": expire, "type": "jwt"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def generate_api_key() -> str:
    return f"ak_{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def api_key_expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=API_KEY_EXPIRE_DAYS)


def validate_scopes(requested: list[str]) -> list[str]:
    invalid = set(requested) - ALL_SCOPES
    if invalid:
        raise ValueError(f"Invalid scopes: {', '.join(sorted(invalid))}")
    return requested
