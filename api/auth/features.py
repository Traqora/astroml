"""Per-account feature gating for the V1 API — issue #1230.

Usage::

    from api.auth.features import require_account_feature

    @router.get("/health/db")
    async def database_health_check(
        _account: Union[ApiAccount, AccountFeatureContext] = Depends(
            require_account_feature("health")
        ),
    ):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.auth.config import is_auth_enabled
from api.auth.dependencies import AuthContext, get_current_auth
from api.database import get_sync_db
from api.models.orm import ApiAccount

KNOWN_FEATURES: frozenset[str] = frozenset(
    {
        "health",
        "model_readiness",
        "baseline_predictions",
    }
)


@dataclass
class AccountFeatureContext:
    """Lightweight stand-in returned when AUTH_ENABLED=false.

    A plain dataclass rather than a transient ORM object so that any
    code reading .id receives an explicit None rather than a SQLAlchemy
    attribute sentinel.  Routes must not commit this object.
    """

    public_key: str
    enabled_features: list
    id: Optional[int] = None


def require_account_feature(feature_name: str):
    """Return a FastAPI dependency that enforces feature access for an account.

    Raises ValueError at factory-call time (import / startup) if feature_name
    is not in KNOWN_FEATURES, so a typo in a route file causes a startup error
    rather than silently denying every request at runtime.

    Uses an explicit ``if / raise`` rather than ``assert`` because Python strips
    ``assert`` statements when invoked with ``-O`` / ``-OO`` / ``PYTHONOPTIMIZE``.
    """
    if feature_name not in KNOWN_FEATURES:
        raise ValueError(
            f"Unknown feature_name {feature_name!r} passed to require_account_feature; "
            f"valid names: {sorted(KNOWN_FEATURES)}"
        )

    def _checker(
        auth: AuthContext = Depends(get_current_auth),
        db: Session = Depends(get_sync_db),
    ) -> Union[ApiAccount, AccountFeatureContext]:
        # When AUTH_ENABLED=false bypass all checks and return a permissive
        # stand-in so tests that don't focus on feature gating work without
        # seeding an ApiAccount row.
        if not is_auth_enabled():
            return AccountFeatureContext(
                public_key="anonymous",
                enabled_features=list(KNOWN_FEATURES),
            )

        if auth.user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "AUTH_REQUIRED",
                    "message": "Authentication required.",
                    "details": {},
                },
            )

        account: Optional[ApiAccount] = db.scalar(
            select(ApiAccount).where(ApiAccount.user_id == auth.user_id)
        )

        if account is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ACCOUNT_NOT_FOUND",
                    "message": f"No account found for user '{auth.subject}'.",
                    "details": {"subject": auth.subject},
                },
            )

        enabled: list[str] = account.enabled_features or []
        if feature_name not in enabled:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "FEATURE_NOT_ENABLED",
                    "message": (
                        f"The '{feature_name}' feature is not enabled "
                        f"for account '{account.public_key}'."
                    ),
                    "details": {
                        "feature": feature_name,
                        "enabled_features": enabled,
                    },
                },
            )

        return account

    return _checker
