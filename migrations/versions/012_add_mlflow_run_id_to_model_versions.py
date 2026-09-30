"""Add mlflow_run_id to model_versions.

Revision ID: 012
Revises: 011
Create Date: 2026-08-31

Closes #764 — store originating MLflow run id on each registry version.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column("mlflow_run_id", sa.String(128), nullable=True),
    )
    op.create_index(
        "ix_model_versions_mlflow_run_id",
        "model_versions",
        ["mlflow_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_versions_mlflow_run_id", table_name="model_versions")
    op.drop_column("model_versions", "mlflow_run_id")
