"""Automated feature engineering and entity set construction using Featuretools.

Provides relational entity set definition, semantic type mapping, and automated
pipeline integration for Stellar and financial transaction networks.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import featuretools as ft
    import woodwork.logical_types as ltypes

    _HAS_FEATURETOOLS = True
except ImportError:
    _HAS_FEATURETOOLS = False
    ft = None
    ltypes = None

logger = logging.getLogger(__name__)


class EntitySetBuilder:
    """Builder for Featuretools EntitySet objects from tabular / relational datasets.

    Supports custom logical types, semantic tags, time indexes, and foreign-key
    relational schemas.
    """

    def __init__(self, id: str = "astroml_entityset") -> None:
        if not _HAS_FEATURETOOLS or ft is None:
            raise ImportError(
                "Featuretools is required for automated feature engineering. "
                "Install it with: pip install featuretools"
            )
        self.entityset = ft.EntitySet(id=id)

    def add_dataframe(
        self,
        name: str,
        dataframe: pd.DataFrame,
        index: str,
        time_index: str | None = None,
        logical_types: dict[str, Any] | None = None,
        semantic_tags: dict[str, Any] | None = None,
    ) -> EntitySetBuilder:
        """Add a dataframe table into the EntitySet.

        Parameters
        ----------
        name : str
            Unique table identifier (e.g. 'accounts', 'transactions').
        dataframe : pd.DataFrame
            Underlying tabular data.
        index : str
            Primary key column name.
        time_index : str | None
            Column representing event timestamp for temporal synthesis.
        logical_types : dict[str, Any] | None
            Custom Woodwork logical types per column.
        semantic_tags : dict[str, Any] | None
            Semantic tag annotations per column.

        Returns
        -------
        EntitySetBuilder
            Self instance for fluent method chaining.
        """
        df = dataframe.copy()
        if time_index and time_index in df.columns:
            df[time_index] = pd.to_datetime(df[time_index], errors="coerce")

        self.entityset.add_dataframe(
            dataframe_name=name,
            dataframe=df,
            index=index,
            time_index=time_index,
            logical_types=logical_types,
            semantic_tags=semantic_tags,
        )
        return self

    def add_relationship(
        self,
        parent_dataframe_name: str,
        parent_column: str,
        child_dataframe_name: str,
        child_column: str,
    ) -> EntitySetBuilder:
        """Add a foreign key relationship between two tables."""
        self.entityset.add_relationship(
            parent_dataframe_name=parent_dataframe_name,
            parent_column_name=parent_column,
            child_dataframe_name=child_dataframe_name,
            child_column_name=child_column,
        )
        return self

    def build(self) -> ft.EntitySet:
        """Return the completed EntitySet."""
        return self.entityset


def build_transaction_entityset(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame | None = None,
    assets_df: pd.DataFrame | None = None,
    entityset_id: str = "transaction_network",
) -> ft.EntitySet:
    """Build a relational EntitySet tailored for financial transaction analysis.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Table with transaction records (must have source/from, destination/to, amount, timestamp).
    accounts_df : pd.DataFrame | None
        Optional table with account metadata (account_id, creation_date, account_type).
    assets_df : pd.DataFrame | None
        Optional table with asset metadata (asset_code, issuer).
    entityset_id : str
        ID for the entityset.

    Returns
    -------
    ft.EntitySet
        Constructed relational EntitySet.
    """
    if not _HAS_FEATURETOOLS or ft is None:
        raise ImportError("Featuretools is required to build transaction entitysets.")

    tx_df = transactions_df.copy()

    # Normalize column names
    col_mapping = {
        "from": "source_account",
        "to": "destination_account",
        "src": "source_account",
        "dst": "destination_account",
        "id": "transaction_id",
        "tx_id": "transaction_id",
        "time": "timestamp",
        "created_at": "timestamp",
    }
    for old_col, new_col in col_mapping.items():
        if old_col in tx_df.columns and new_col not in tx_df.columns:
            tx_df.rename(columns={old_col: new_col}, inplace=True)

    if "transaction_id" not in tx_df.columns:
        tx_df["transaction_id"] = [f"tx_{i}" for i in range(len(tx_df))]

    if "timestamp" in tx_df.columns:
        tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"], errors="coerce")
    else:
        tx_df["timestamp"] = pd.date_range("2026-01-01", periods=len(tx_df), freq="min")

    # Generate accounts table if not provided
    if accounts_df is None:
        src_accounts = tx_df["source_account"].dropna().unique()
        dst_accounts = tx_df["destination_account"].dropna().unique()
        all_accounts = sorted(list(set(src_accounts).union(set(dst_accounts))))
        accounts_df = pd.DataFrame({"account_id": all_accounts})
    else:
        accounts_df = accounts_df.copy()
        if "account_id" not in accounts_df.columns and "id" in accounts_df.columns:
            accounts_df.rename(columns={"id": "account_id"}, inplace=True)

    builder = EntitySetBuilder(id=entityset_id)
    builder.add_dataframe(
        name="accounts",
        dataframe=accounts_df,
        index="account_id",
    )

    builder.add_dataframe(
        name="transactions",
        dataframe=tx_df,
        index="transaction_id",
        time_index="timestamp",
    )

    # Link accounts to transactions (source and destination)
    builder.add_relationship(
        parent_dataframe_name="accounts",
        parent_column="account_id",
        child_dataframe_name="transactions",
        child_column="source_account",
    )

    if assets_df is not None and "asset_code" in tx_df.columns:
        builder.add_dataframe(
            name="assets",
            dataframe=assets_df,
            index="asset_code",
        )
        builder.add_relationship(
            parent_dataframe_name="assets",
            parent_column="asset_code",
            child_dataframe_name="transactions",
            child_column="asset_code",
        )

    return builder.build()
