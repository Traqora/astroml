"""Initial raw data storage schema.

Revision ID: 001
Revises: None
Create Date: 2026-02-25

Creates the five core tables for raw Stellar blockchain data:
ledgers, transactions, operations, accounts, assets.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- ledgers ---------------------------------------------------------------
    op.create_table(
        "ledgers",
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column("prev_hash", sa.String(64), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "successful_transaction_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "failed_transaction_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "operation_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("total_coins", sa.Numeric(), nullable=True),
        sa.Column("fee_pool", sa.Numeric(), nullable=True),
        sa.Column("base_fee_in_stroops", sa.Integer(), nullable=True),
        sa.Column("protocol_version", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("sequence"),
        sa.UniqueConstraint("hash"),
    )
    op.create_index("ix_ledgers_closed_at", "ledgers", ["closed_at"])

    # -- transactions ----------------------------------------------------------
    op.create_table(
        "transactions",
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column("ledger_sequence", sa.Integer(), nullable=False),
        sa.Column("source_account", sa.String(56), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fee", sa.BigInteger(), nullable=False),
        sa.Column("operation_count", sa.SmallInteger(), nullable=False),
        sa.Column("successful", sa.Boolean(), nullable=False),
        sa.Column("memo_type", sa.String(16), nullable=True),
        sa.Column("memo", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("hash"),
        sa.ForeignKeyConstraint(["ledger_sequence"], ["ledgers.sequence"]),
    )
    op.create_index(
        "ix_transactions_source_account_created_at",
        "transactions",
        ["source_account", "created_at"],
    )
    op.create_index(
        "ix_transactions_ledger_sequence", "transactions", ["ledger_sequence"]
    )

    # -- operations ------------------------------------------------------------
    op.create_table(
        "operations",
        sa.Column(
            "id", sa.BigInteger(), nullable=False, autoincrement=True
        ),
        sa.Column("transaction_hash", sa.String(64), nullable=False),
        sa.Column("application_order", sa.SmallInteger(), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("source_account", sa.String(56), nullable=False),
        sa.Column("destination_account", sa.String(56), nullable=True),
        sa.Column("amount", sa.Numeric(), nullable=True),
        sa.Column("asset_code", sa.String(12), nullable=True),
        sa.Column("asset_issuer", sa.String(56), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["transaction_hash"], ["transactions.hash"]),
    )
    op.create_index(
        "ix_operations_source_created_at",
        "operations",
        ["source_account", "created_at"],
    )
    op.create_index(
        "ix_operations_dest_created_at",
        "operations",
        ["destination_account", "created_at"],
        postgresql_where=sa.text("destination_account IS NOT NULL"),
    )
    op.create_index(
        "ix_operations_transaction_hash", "operations", ["transaction_hash"]
    )
    op.create_index("ix_operations_type", "operations", ["type"])

    # -- accounts --------------------------------------------------------------
    op.create_table(
        "accounts",
        sa.Column("account_id", sa.String(56), nullable=False),
        sa.Column("balance", sa.Numeric(), nullable=True),
        sa.Column("sequence", sa.BigInteger(), nullable=True),
        sa.Column("home_domain", sa.String(32), nullable=True),
        sa.Column("flags", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_modified_ledger", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("account_id"),
    )
    op.create_index("ix_accounts_updated_at", "accounts", ["updated_at"])

    # -- assets ----------------------------------------------------------------
    op.create_table(
        "assets",
        sa.Column(
            "id", sa.Integer(), nullable=False, autoincrement=True
        ),
        sa.Column("asset_type", sa.String(16), nullable=False),
        sa.Column("asset_code", sa.String(12), nullable=False),
        sa.Column("asset_issuer", sa.String(56), nullable=True),
        sa.Column("first_seen_ledger", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_assets_code_issuer",
        "assets",
        ["asset_code", sa.text("COALESCE(asset_issuer, '')")],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("assets")
    op.drop_table("accounts")
    op.drop_table("operations")
    op.drop_table("transactions")
    op.drop_table("ledgers")
