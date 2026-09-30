"""Add effects table for granular account state changes.

Revision ID: 003
Revises: 002
Create Date: 2026-04-24

Creates the effects table to capture granular state changes from operations.
Effects represent outcomes of operations on accounts, such as balance
changes, signers, flags, and other state modifications. This table provides
a more granular view of account state changes than operations alone.

The effects table is referenced by the enhanced streaming service but was missing
from the database schema.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- effects ---------------------------------------------------------------
    op.create_table(
        "effects",
        sa.Column("id", sa.BigInteger(), nullable=False, autoincrement=True),
        sa.Column("account", sa.String(56), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("amount", sa.Numeric(), nullable=True),
        sa.Column("asset_code", sa.String(12), nullable=True),
        sa.Column("asset_issuer", sa.String(56), nullable=True),
        sa.Column("destination_account", sa.String(56), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for performance
    op.create_index("ix_effects_account_created_at", "effects", ["account", "created_at"])
    op.create_index("ix_effects_type_created_at", "effects", ["type", "created_at"])
    op.create_index(
        "ix_effects_destination_created_at",
        "effects",
        ["destination_account", "created_at"],
        postgresql_where=sa.text("destination_account IS NOT NULL"),
    )
    op.create_index("ix_effects_account", "effects", ["account"])
    op.create_index("ix_effects_type", "effects", ["type"])
    op.create_index("ix_effects_created_at", "effects", ["created_at"])


def downgrade() -> None:
    # Drop indexes first
    op.drop_index("ix_effects_created_at", table_name="effects")
    op.drop_index("ix_effects_type", table_name="effects")
    op.drop_index("ix_effects_account", table_name="effects")
    op.drop_index("ix_effects_destination_created_at", table_name="effects")
    op.drop_index("ix_effects_type_created_at", table_name="effects")
    op.drop_index("ix_effects_account_created_at", table_name="effects")

    # Drop the table
    op.drop_table("effects")
