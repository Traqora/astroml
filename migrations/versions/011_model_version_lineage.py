"""Model version lineage — add lineage column to model_versions.

Revision ID: 011
Revises: 010
Create Date: 2026-08-29

Closes #718 — record activation/rollback transitions with lineage.

The registry recorded serving transitions by assigning to ``ModelVersion.metadata``.
That attribute is SQLAlchemy's reserved ``MetaData`` object, not a mapped column,
so nothing was ever persisted. This adds a real column for it.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "011"
down_revision: Union[str, None] = "010a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column(
            "lineage",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("model_versions", "lineage")
