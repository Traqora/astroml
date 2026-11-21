"""Add user_id and enabled_features to api_accounts.

Revision ID: 015
Revises: 014
Create Date: 2026-11-21

Adds user_id (nullable BigInteger/Integer, unique) and enabled_features
(nullable JSON/JSONB) columns to the api_accounts table, plus an index on
user_id so look-ups by authenticated user are fast.

Relates to issue #1230 — return useful client errors for invalid account
features.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("api_accounts", sa.Column("user_id", _ID, nullable=True))
    op.add_column("api_accounts", sa.Column("enabled_features", _JSON, nullable=True))
    op.create_index("ix_api_accounts_user_id", "api_accounts", ["user_id"])
    op.create_unique_constraint("uq_api_accounts_user_id", "api_accounts", ["user_id"])


def downgrade() -> None:
    op.drop_constraint("uq_api_accounts_user_id", "api_accounts", type_="unique")
    op.drop_index("ix_api_accounts_user_id", table_name="api_accounts")
    op.drop_column("api_accounts", "enabled_features")
    op.drop_column("api_accounts", "user_id")
