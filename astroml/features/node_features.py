from __future__ import annotations

"""
Base node features from transaction graph.

Features per node:
- in_degree / out_degree
- total_received / total_sent (volume)
- account_age: seconds since first_seen_ts relative to a provided reference time

Inputs:
- edges: iterable of dict-like with keys: src, dst, amount, timestamp (ints/floats acceptable)
- nodes: optional mapping node_id -> first_seen_ts (epoch seconds). If not provided, age is computed
         from the minimum timestamp observed in edges for that node.
- ref_time: reference timestamp (epoch seconds) to compute age. Defaults to max timestamp in edges.

Returns:
- pandas.DataFrame indexed by node id with columns:
  ['in_degree','out_degree','total_received','total_sent','account_age']
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Hashable
import pandas as pd
import numpy as np

Edge = Dict[str, object]


from astroml.features.asset_diversity import compute_asset_diversity

def compute_node_features(
    edges: Iterable[Edge],
    nodes_first_seen: Optional[Dict[Hashable, float]] = None,
    ref_time: Optional[float] = None,
) -> pd.DataFrame:
    rows_src = []
    rows_dst = []
    
    # Track asset interactions for diversity metric
    asset_rows = []
    
    max_ts = -np.inf

    for e in edges:
        src = e.get('src')
        dst = e.get('dst')
        amt = float(e.get('amount', 0.0) or 0.0)
        ts = float(e.get('timestamp', 0.0) or 0.0)
        asset = e.get('asset', 'UNKNOWN')
        
        max_ts = max(max_ts, ts)
        if src is not None:
            rows_src.append((src, amt, ts))
            asset_rows.append((src, asset))
            
        if dst is not None:
            rows_dst.append((dst, amt, ts))
            asset_rows.append((dst, asset))

    # Build DataFrames for aggregation
    src_df = pd.DataFrame(rows_src, columns=['node','amount','timestamp'])
    dst_df = pd.DataFrame(rows_dst, columns=['node','amount','timestamp'])
    asset_df = pd.DataFrame(asset_rows, columns=['node', 'asset'])

    if ref_time is None:
        ref_time = float(max_ts if max_ts != -np.inf else 0.0)

    # Degrees
    out_degree = src_df.groupby('node').size().astype(int).rename('out_degree') if not src_df.empty else pd.Series(dtype=int, name='out_degree')
    in_degree = dst_df.groupby('node').size().astype(int).rename('in_degree') if not dst_df.empty else pd.Series(dtype=int, name='in_degree')

    # Volumes
    total_sent = src_df.groupby('node')['amount'].sum().rename('total_sent') if not src_df.empty else pd.Series(dtype=float, name='total_sent')
    total_received = dst_df.groupby('node')['amount'].sum().rename('total_received') if not dst_df.empty else pd.Series(dtype=float, name='total_received')

    # First seen from edges
    first_seen_src = src_df.groupby('node')['timestamp'].min().rename('first_seen_src') if not src_df.empty else pd.Series(dtype=float, name='first_seen_src')
    first_seen_dst = dst_df.groupby('node')['timestamp'].min().rename('first_seen_dst') if not dst_df.empty else pd.Series(dtype=float, name='first_seen_dst')

    first_seen_edge = pd.concat([first_seen_src, first_seen_dst], axis=1).min(axis=1).rename('first_seen_edge')
    
    # Asset Diversity
    if not asset_df.empty:
        # Group by node and asset to get counts
        asset_counts = asset_df.groupby(['node', 'asset']).size()
        
        # Apply diversity function per node
        diversity_df = asset_counts.groupby('node').apply(
            lambda x: pd.Series(compute_asset_diversity(x.droplevel('node')))
        ).unstack()
        diversity_df = diversity_df.fillna(0)
    else:
        diversity_df = pd.DataFrame(columns=['unique_asset_count', 'asset_entropy'])

    # Merge all
    feats = pd.concat([
        in_degree, out_degree, total_received, total_sent, 
        first_seen_edge, diversity_df
    ], axis=1).fillna(0)

    # If external first_seen provided, prefer it where available
    if nodes_first_seen is not None and len(nodes_first_seen) > 0:
        provided = pd.Series(nodes_first_seen, name='first_seen_provided', dtype=float)
        feats['first_seen'] = feats['first_seen_edge']
        feats.loc[feats.index.intersection(provided.index), 'first_seen'] = provided.loc[feats.index.intersection(provided.index)]
    else:
        feats['first_seen'] = feats['first_seen_edge']

    # Account age: ref_time - first_seen; clamp at 0
    feats['account_age'] = (float(ref_time) - feats['first_seen']).clip(lower=0.0)

    # Finalize columns and dtypes
    feats = feats.drop(columns=['first_seen_edge'])
    feats['in_degree'] = feats['in_degree'].astype(int)
    feats['out_degree'] = feats['out_degree'].astype(int)
    feats['total_received'] = feats['total_received'].astype(float)
    feats['total_sent'] = feats['total_sent'].astype(float)
    feats['account_age'] = feats['account_age'].astype(float)
    
    if 'unique_asset_count' in feats.columns:
        feats['unique_asset_count'] = feats['unique_asset_count'].astype(int)
        feats['asset_entropy'] = feats['asset_entropy'].astype(float)
    else:
        feats['unique_asset_count'] = 0
        feats['asset_entropy'] = 0.0

    # Ensure nodes that appear only in nodes_first_seen (no edges) are included
    if nodes_first_seen is not None and len(nodes_first_seen) > 0:
        provided = pd.Series(nodes_first_seen, name='first_seen_provided', dtype=float)
        missing = provided.index.difference(feats.index)
        if len(missing) > 0:
            extra = pd.DataFrame(index=missing)
            extra['in_degree'] = 0
            extra['out_degree'] = 0
            extra['total_received'] = 0.0
            extra['total_sent'] = 0.0
            extra['unique_asset_count'] = 0
            extra['asset_entropy'] = 0.0
            extra['first_seen'] = provided.loc[missing]
            extra['account_age'] = (float(ref_time) - extra['first_seen']).clip(lower=0.0)
            feats = pd.concat([feats, extra])

    # Order columns
    feats = feats[['in_degree','out_degree','total_received','total_sent','account_age','first_seen', 'unique_asset_count', 'asset_entropy']]

    return feats.sort_index()
