"""Add models and model_versions registry tables.

Revision ID: 010a
Revises: 010
Create Date: 2026-09-26

The ORM registry (``astroml.db.models.DbModel`` / ``ModelVersion``) was
introduced without a matching migration, so a database built from
``alembic upgrade head`` never got these tables and revision 011 died with
``relation "model_versions" does not exist``. Deployments that created them
via ``Base.metadata.create_all`` are unaffected: creation is skipped when
the table already exists.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "010a"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _table_exists("models"):
        op.create_table(
            "models",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("framework", sa.String(length=32), nullable=False),
            sa.Column("task_type", sa.String(length=32), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.UniqueConstraint("name", name="uq_models_name"),
            sa.CheckConstraint(
                "framework IN ('pytorch', 'tensorflow', 'sklearn', 'xgboost', 'lightgbm', 'custom')",
                name="ck_models_framework",
            ),
            sa.CheckConstraint(
                "task_type IN ('classification', 'regression', 'anomaly_detection', "
                "'clustering', 'custom')",
                name="ck_models_task_type",
            ),
        )
        op.create_index("ix_models_framework", "models", ["framework"])
        op.create_index("ix_models_task_type", "models", ["task_type"])
        op.create_index("ix_models_is_active", "models", ["is_active"])

    if not _table_exists("model_versions"):
        op.create_table(
            "model_versions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("model_id", sa.Integer(), nullable=False),
            sa.Column("version", sa.String(length=32), nullable=False),
            sa.Column("artifact_path", sa.String(length=512), nullable=False),
            sa.Column(
                "hyperparameters",
                sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
                nullable=True,
            ),
            sa.Column(
                "metrics",
                sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
                nullable=True,
            ),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="training"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("deployed_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["model_id"], ["models.id"], name="fk_model_versions_model_id"),
            sa.UniqueConstraint("model_id", "version", name="uq_model_versions_model_version"),
            sa.CheckConstraint(
                "status IN ('training', 'trained', 'deployed', 'archived', 'failed')",
                name="ck_model_versions_status",
            ),
        )
        op.create_index("ix_model_versions_model_id", "model_versions", ["model_id"])
        op.create_index("ix_model_versions_status", "model_versions", ["status"])
        op.create_index("ix_model_versions_created_at", "model_versions", ["created_at"])


def downgrade() -> None:
    if _table_exists("model_versions"):
        op.drop_index("ix_model_versions_created_at", table_name="model_versions")
        op.drop_index("ix_model_versions_status", table_name="model_versions")
        op.drop_index("ix_model_versions_model_id", table_name="model_versions")
        op.drop_table("model_versions")
    if _table_exists("models"):
        op.drop_index("ix_models_is_active", table_name="models")
        op.drop_index("ix_models_task_type", table_name="models")
        op.drop_index("ix_models_framework", table_name="models")
        op.drop_table("models")
