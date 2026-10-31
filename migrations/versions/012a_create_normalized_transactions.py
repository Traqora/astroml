"""Create the normalized_transactions table.

Revision ID: 012a
Revises: 012
Create Date: 2026-09-25

The ORM model (``astroml.db.models.NormalizedTransaction``) was introduced
without a matching migration, the same gap already fixed once for
``models``/``model_versions`` in revision 010a. Revision 013 adds
``ledger_sequence``/``operation_id``/``hop_index`` columns and their unique
constraint to this table, and revision 014 adds a composite index on top of
those columns — both assume the table already exists, so
``alembic upgrade head`` on a fresh database has always died at revision 013
with ``relation "normalized_transactions" does not exist``.

Mirrors 010a's guard: creation is skipped when the table already exists, so
a deployment that created it via ``Base.metadata.create_all`` directly is
unaffected.

This creates the table exactly as it stood before revision 013's columns:
``id``, ``transaction_hash``, ``sender``, ``receiver``, ``asset``,
``amount``, ``timestamp``, plus the indexes 013/014 do not themselves
create (``ix_normalized_transactions_hash``,
``ix_normalized_transactions_sender_timestamp``,
``ix_normalized_transactions_receiver_timestamp``). 013 and 014 remain
responsible for the columns/indexes they already add.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012a"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _table_exists("normalized_transactions"):
        return

    op.create_table(
        "normalized_transactions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("transaction_hash", sa.String(length=64), nullable=False),
        sa.Column("sender", sa.String(length=56), nullable=False),
        sa.Column("receiver", sa.String(length=56), nullable=True),
        sa.Column("asset", sa.String(length=70), nullable=False),
        sa.Column("amount", sa.Numeric(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_normalized_transactions_hash", "normalized_transactions", ["transaction_hash"]
    )
    op.create_index(
        "ix_normalized_transactions_sender_timestamp",
        "normalized_transactions",
        ["sender", "timestamp"],
    )
    op.create_index(
        "ix_normalized_transactions_receiver_timestamp",
        "normalized_transactions",
        ["receiver", "timestamp"],
        postgresql_where=sa.text("receiver IS NOT NULL"),
    )


def downgrade() -> None:
    if not _table_exists("normalized_transactions"):
        return
    op.drop_index(
        "ix_normalized_transactions_receiver_timestamp", table_name="normalized_transactions"
    )
    op.drop_index(
        "ix_normalized_transactions_sender_timestamp", table_name="normalized_transactions"
    )
    op.drop_index("ix_normalized_transactions_hash", table_name="normalized_transactions")
    op.drop_table("normalized_transactions")
