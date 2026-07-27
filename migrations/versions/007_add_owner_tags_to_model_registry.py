"""Add owner and tags to model registry.

Revision ID: 007
Revises: 006
Create Date: 2026-06-26

Adds owner and tags fields to model_registry table
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("model_registry", sa.Column("owner", sa.String(128), nullable=True))
    op.add_column("model_registry", sa.Column("tags", _JSON, nullable=True))
    op.create_index("ix_model_registry_owner", "model_registry", ["owner"])


def downgrade() -> None:
    op.drop_index("ix_model_registry_owner", table_name="model_registry")
    op.drop_column("model_registry", "tags")
    op.drop_column("model_registry", "owner")
