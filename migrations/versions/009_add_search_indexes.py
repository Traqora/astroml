"""Add search optimization indexes.

Revision ID: 009
Revises: 008
Create Date: 2026-06-26

Adds indexes for name and version to optimize search
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_model_registry_name", "model_registry", ["name"])
    op.create_index("ix_model_registry_version", "model_registry", ["version"])


def downgrade() -> None:
    op.drop_index("ix_model_registry_name", table_name="model_registry")
    op.drop_index("ix_model_registry_version", table_name="model_registry")
