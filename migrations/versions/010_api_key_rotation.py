"""API key rotation — add overlap columns to api_keys.

Revision ID: 010
Revises: 009
Create Date: 2026-07-27

Closes #534 — API Key Rotation & Revocation
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # overlap_key_hash: stores the SHA-256 hash of the rotated-out (old) key
    # so it remains authenticatable during the overlap window.
    op.add_column(
        "api_keys",
        sa.Column(
            "overlap_key_hash",
            sa.String(64),
            nullable=True,
        ),
    )
    op.add_column(
        "api_keys",
        sa.Column(
            "overlap_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_index("ix_api_keys_overlap_key_hash", "api_keys", ["overlap_key_hash"])


def downgrade() -> None:
    op.drop_index("ix_api_keys_overlap_key_hash", table_name="api_keys")
    op.drop_column("api_keys", "overlap_expires_at")
    op.drop_column("api_keys", "overlap_key_hash")
