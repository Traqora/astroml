from astroml.utils.exceptions import AstroMLError
"""Offline Feature Store for batch feature storage and point-in-time correct joins.

Provides analytical, time-travel, and training dataset extraction with
point-in-time correctness to prevent data leakage in ML training workflows.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PointInTimeJoinConfig:
    """Configuration for point-in-time feature join."""

    entity_col: str = "entity_id"
    timestamp_col: str = "timestamp"
    feature_timestamp_col: str = "computed_at"
    lookback_window_seconds: int | None = None
    allow_exact_match: bool = True


class OfflineFeatureStore:
    """Offline storage and point-in-time feature join engine."""

    def __init__(self, storage_path: str | Path = "./offline_feature_store") -> None:
        """Initialize offline feature store."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._parquet_dir = self.storage_path / "parquet"
        self._parquet_dir.mkdir(parents=True, exist_ok=True)

    def _feature_path(self, feature_name: str) -> Path:
        """Get directory path for partitioned feature parquet files."""
        p = self._parquet_dir / feature_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_offline_features(
        self,
        feature_name: str,
        df: pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        partition_by_date: bool = True,
    ) -> int:
        """Write feature values to offline Parquet storage with partitioning."""
        if df.empty:
            return 0

        working_df = df.copy()
        if entity_col not in working_df.columns:
            if working_df.index.name == entity_col or not working_df.index.empty:
                working_df[entity_col] = working_df.index

        if timestamp_col not in working_df.columns:
            working_df[timestamp_col] = pd.to_datetime(datetime.now(timezone.utc))
        else:
            working_df[timestamp_col] = pd.to_datetime(working_df[timestamp_col])

        feat_dir = self._feature_path(feature_name)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        file_path = feat_dir / f"batch_{timestamp_str}.parquet"

        working_df.to_parquet(file_path, index=False)
        logger.info("Wrote %d rows for feature %s to %s", len(working_df), feature_name, file_path)
        return len(working_df)

    def read_offline_feature(
        self,
        feature_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        entity_ids: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read historical feature data for given time range and entities."""
        feat_dir = self._feature_path(feature_name)
        files = list(feat_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as exc:
                logger.warning("Could not read parquet file %s: %s", f, exc)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        if "timestamp" in combined.columns:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            if start_time is not None:
                combined = combined[combined["timestamp"] >= pd.to_datetime(start_time)]
            if end_time is not None:
                combined = combined[combined["timestamp"] <= pd.to_datetime(end_time)]

        if entity_ids is not None and "entity_id" in combined.columns:
            combined = combined[
                combined["entity_id"].astype(str).isin([str(e) for e in entity_ids])
            ]

        return combined

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_names: Sequence[str],
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        lookback_seconds: int | None = None,
    ) -> pd.DataFrame:
        """Perform point-in-time (as-of) join to build leak-free training datasets.

        For each entity record at `timestamp`, finds the latest feature value computed
        BEFORE or AT that timestamp.
        """
        if entity_df.empty:
            return entity_df.copy()

        result_df = entity_df.copy()
        result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
        result_df = result_df.sort_values(timestamp_col)

        for feat_name in feature_names:
            feat_df = self.read_offline_feature(feat_name)
            if feat_df.empty:
                # Fill missing feature column with NaN
                val_col = [c for c in feat_df.columns if c not in ("entity_id", "timestamp")]
                target_col = val_col[0] if val_col else feat_name
                result_df[target_col] = np.nan
                continue

            feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])
            feat_df = feat_df.sort_values("timestamp")

            val_cols = [c for c in feat_df.columns if c not in ("entity_id", "timestamp")]
            target_col = val_cols[0] if val_cols else feat_name

            # Merge as-of for point-in-time correctness
            joined = pd.merge_asof(
                result_df,
                feat_df[[entity_col, "timestamp", target_col]],
                left_on=timestamp_col,
                right_on="timestamp",
                by=entity_col,
                direction="backward",
                suffixes=("", "_feat"),
            )
            # Remove duplicated timestamp column if created
            if "timestamp_feat" in joined.columns:
                joined = joined.drop(columns=["timestamp_feat"])
            result_df = joined

        return result_df

    def get_feature_statistics(
        self,
        feature_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Compute statistical summary and monitoring metrics for an offline feature."""
        df = self.read_offline_feature(feature_name, start_time, end_time)
        if df.empty:
            return {"feature_name": feature_name, "count": 0, "status": "no_data"}

        val_cols = [c for c in df.columns if c not in ("entity_id", "timestamp")]
        if not val_cols:
            return {"feature_name": feature_name, "count": len(df)}

        target_col = val_cols[0]
        series = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if series.empty:
            return {
                "feature_name": feature_name,
                "count": len(df),
                "null_count": int(df[target_col].isna().sum()),
                "unique_values": int(df[target_col].nunique()),
            }

        return {
            "feature_name": feature_name,
            "count": int(len(series)),
            "null_count": int(df[target_col].isna().sum()),
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
        }


def create_offline_store(
    storage_path: str | Path = "./offline_feature_store",
) -> OfflineFeatureStore:
    """Factory function for OfflineFeatureStore."""
    return OfflineFeatureStore(storage_path=storage_path)
