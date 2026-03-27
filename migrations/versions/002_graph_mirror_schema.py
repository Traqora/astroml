"""Add normalized PostgreSQL graph mirror schema.

Revision ID: 002
Revises: 001
Create Date: 2026-03-24

Adds a normalized graph mirror alongside the existing raw Stellar tables:

- graph_accounts: canonical account nodes
- graph_edges: shared directed edge/event rows
- graph_transaction_details: transaction-only attributes
- graph_claim_details: claim-only attributes
- graph_payment_details: payment-only attributes
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graph_accounts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("account_address", sa.String(length=56), nullable=False),
        sa.Column("account_type", sa.String(length=32), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("account_address"),
    )
    op.create_index(
        "ix_graph_accounts_last_seen_at",
        "graph_accounts",
        ["last_seen_at"],
    )
    op.create_index(
        "ix_graph_accounts_account_type",
        "graph_accounts",
        ["account_type"],
    )

    op.create_table(
        "graph_edges",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("edge_type", sa.String(length=16), nullable=False),
        sa.Column("source_account_id", sa.BigInteger(), nullable=False),
        sa.Column("destination_account_id", sa.BigInteger(), nullable=True),
        sa.Column("asset_id", sa.Integer(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ledger_sequence", sa.Integer(), nullable=True),
        sa.Column("event_index", sa.Integer(), nullable=True),
        sa.Column("transaction_hash", sa.String(length=64), nullable=True),
        sa.Column("external_event_id", sa.String(length=128), nullable=False),
        sa.Column("amount", sa.Numeric(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "edge_type IN ('transaction', 'claim', 'payment')",
            name="ck_graph_edges_edge_type",
        ),
        sa.CheckConstraint(
            "source_account_id <> destination_account_id OR destination_account_id IS NULL",
            name="ck_graph_edges_distinct_accounts",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.id"],
        ),
        sa.ForeignKeyConstraint(
            ["destination_account_id"],
            ["graph_accounts.id"],
        ),
        sa.ForeignKeyConstraint(
            ["source_account_id"],
            ["graph_accounts.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "edge_type",
            "external_event_id",
            name="uq_graph_edges_type_external_event_id",
        ),
        sa.UniqueConstraint(
            "id",
            "edge_type",
            name="uq_graph_edges_id_edge_type",
        ),
    )
    op.create_index("ix_graph_edges_occurred_at", "graph_edges", ["occurred_at"])
    op.create_index(
        "ix_graph_edges_source_occurred_at",
        "graph_edges",
        ["source_account_id", "occurred_at"],
    )
    op.create_index(
        "ix_graph_edges_destination_occurred_at",
        "graph_edges",
        ["destination_account_id", "occurred_at"],
    )
    op.create_index(
        "ix_graph_edges_type_occurred_at",
        "graph_edges",
        ["edge_type", "occurred_at"],
    )
    op.create_index(
        "ix_graph_edges_asset_occurred_at",
        "graph_edges",
        ["asset_id", "occurred_at"],
    )
    op.create_index(
        "ix_graph_edges_status_occurred_at",
        "graph_edges",
        ["status", "occurred_at"],
    )
    op.create_index(
        "ix_graph_edges_tx_hash",
        "graph_edges",
        ["transaction_hash"],
        postgresql_where=sa.text("transaction_hash IS NOT NULL"),
    )
    op.create_index(
        "ix_graph_edges_ledger_event",
        "graph_edges",
        ["ledger_sequence", "event_index"],
    )

    op.create_table(
        "graph_transaction_details",
        sa.Column("edge_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "edge_type",
            sa.String(length=16),
            nullable=False,
            server_default="transaction",
        ),
        sa.Column("successful", sa.Boolean(), nullable=True),
        sa.Column("operation_count", sa.SmallInteger(), nullable=True),
        sa.Column("fee", sa.BigInteger(), nullable=True),
        sa.Column("memo_type", sa.String(length=16), nullable=True),
        sa.Column("memo", sa.Text(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint(
            "edge_type = 'transaction'",
            name="ck_graph_transaction_details_edge_type",
        ),
        sa.ForeignKeyConstraint(
            ["edge_id", "edge_type"],
            ["graph_edges.id", "graph_edges.edge_type"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("edge_id"),
    )

    op.create_table(
        "graph_claim_details",
        sa.Column("edge_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "edge_type",
            sa.String(length=16),
            nullable=False,
            server_default="claim",
        ),
        sa.Column("claim_reference", sa.String(length=128), nullable=True),
        sa.Column("claim_status", sa.String(length=32), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint(
            "edge_type = 'claim'",
            name="ck_graph_claim_details_edge_type",
        ),
        sa.ForeignKeyConstraint(
            ["edge_id", "edge_type"],
            ["graph_edges.id", "graph_edges.edge_type"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("edge_id"),
    )
    op.create_index(
        "ix_graph_claim_details_claim_status",
        "graph_claim_details",
        ["claim_status"],
    )

    op.create_table(
        "graph_payment_details",
        sa.Column("edge_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "edge_type",
            sa.String(length=16),
            nullable=False,
            server_default="payment",
        ),
        sa.Column("payment_reference", sa.String(length=128), nullable=True),
        sa.Column("payment_status", sa.String(length=32), nullable=True),
        sa.Column("fee_amount", sa.Numeric(), nullable=True),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint(
            "edge_type = 'payment'",
            name="ck_graph_payment_details_edge_type",
        ),
        sa.CheckConstraint(
            "fee_amount >= 0 OR fee_amount IS NULL",
            name="ck_graph_payment_details_fee_amount_non_negative",
        ),
        sa.ForeignKeyConstraint(
            ["edge_id", "edge_type"],
            ["graph_edges.id", "graph_edges.edge_type"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("edge_id"),
    )
    op.create_index(
        "ix_graph_payment_details_payment_status",
        "graph_payment_details",
        ["payment_status"],
    )


def downgrade() -> None:
    op.drop_index("ix_graph_payment_details_payment_status", table_name="graph_payment_details")
    op.drop_table("graph_payment_details")
    op.drop_index("ix_graph_claim_details_claim_status", table_name="graph_claim_details")
    op.drop_table("graph_claim_details")
    op.drop_table("graph_transaction_details")
    op.drop_index("ix_graph_edges_ledger_event", table_name="graph_edges")
    op.drop_index("ix_graph_edges_tx_hash", table_name="graph_edges")
    op.drop_index("ix_graph_edges_status_occurred_at", table_name="graph_edges")
    op.drop_index("ix_graph_edges_asset_occurred_at", table_name="graph_edges")
    op.drop_index("ix_graph_edges_type_occurred_at", table_name="graph_edges")
    op.drop_index("ix_graph_edges_destination_occurred_at", table_name="graph_edges")
    op.drop_index("ix_graph_edges_source_occurred_at", table_name="graph_edges")
    op.drop_index("ix_graph_edges_occurred_at", table_name="graph_edges")
    op.drop_table("graph_edges")
    op.drop_index("ix_graph_accounts_account_type", table_name="graph_accounts")
    op.drop_index("ix_graph_accounts_last_seen_at", table_name="graph_accounts")
    op.drop_table("graph_accounts")
