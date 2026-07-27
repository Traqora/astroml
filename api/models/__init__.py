"""API-layer ORM models (issue #251).

All models use the shared ``Base`` from ``astroml.db.schema`` so that
``alembic upgrade head`` creates every table in one pass.
"""
from api.models.orm import (
    ApiAccount,
    ApiTransaction,
    FraudAlert,
    LoyaltyPoints,
    PointsTransaction,
    ModelRegistry,
    User,
    ApiKey,
)

# Aliases for backward compatibility (not registered as separate mappers)
Account = ApiAccount
Transaction = ApiTransaction

__all__ = [
    "ApiAccount",
    "ApiTransaction",
    "Account",
    "Transaction",
    "FraudAlert",
    "LoyaltyPoints",
    "PointsTransaction",
    "ModelRegistry",
    "User",
    "ApiKey",
]
