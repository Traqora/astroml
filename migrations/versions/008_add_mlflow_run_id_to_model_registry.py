"""Add mlflow_run_id to model registry.

Revision ID: 008
Revises: 007
Create Date: 2026-06-26

Adds mlflow_run_id column and index to model_registry table
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("model_registry", sa.Column("mlflow_run_id", sa.String(128), nullable=True))
    op.create_index("ix_model_registry_mlflow_run_id", "model_registry", ["mlflow_run_id"])


def downgrade() -> None:
    op.drop_index("ix_model_registry_mlflow_run_id", table_name="model_registry")
    op.drop_column("model_registry", "mlflow_run_id")
