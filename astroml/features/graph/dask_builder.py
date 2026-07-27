"""Dask-based parallel graph building for large-scale graphs.

This module provides distributed graph building using Dask for graphs
with 1M+ edges where parallel processing provides significant speedup.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

# Check if dask is available (optional dependency)
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning(
        "Dask not installed. Install with: pip install astroml[dask] "
        "to enable parallel graph building."
    )


class DaskGraphBuilder:
    """Parallel graph builder using Dask for large-scale graphs.
    
    Uses Dask for distributed edge processing, suitable for graphs
    with 1M+ edges. Falls back to NetworkX for smaller graphs.
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        threads_per_worker: int = 2,
        memory_limit: str = "4GB"
    ):
        """Initialize Dask graph builder.
        
        Args:
            n_workers: Number of Dask workers (default: CPU count)
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker
        """
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is required for DaskGraphBuilder. "
                "Install with: pip install astroml[dask]"
            )
        
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.client: Optional[Client] = None
    
    def __enter__(self):
        """Start Dask cluster."""
        cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
            silence_logs=logging.WARNING
        )
        self.client = Client(cluster)
        logger.info(f"Started Dask cluster: {self.client.dashboard_link}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop Dask cluster."""
        if self.client:
            self.client.close()
            self.client = None
    
    def build_graph(
        self,
        edges_df: pd.DataFrame,
        source_col: str = "source",
        target_col: str = "target",
        edge_attrs: Optional[List[str]] = None,
        directed: bool = True,
        partition_size: int = 100000
    ) -> nx.Graph:
        """Build graph from edge DataFrame using Dask.
        
        Args:
            edges_df: DataFrame with edge data
            source_col: Source node column
            target_col: Target node column
            edge_attrs: Additional edge attribute columns
            directed: Whether to build directed graph
            partition_size: Number of edges per partition
            
        Returns:
            NetworkX graph
        """
        if not DASK_AVAILABLE or self.client is None:
            raise RuntimeError("Dask client not initialized. Use context manager.")
        
        n_edges = len(edges_df)
        logger.info(f"Building graph with {n_edges:,} edges using Dask")
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(edges_df, npartitions=max(1, n_edges // partition_size))
        
        # Process edges in parallel
        edge_attrs = edge_attrs or []
        required_cols = [source_col, target_col] + edge_attrs
        
        # Extract edge tuples
        def extract_edges(partition):
            """Extract edge tuples from partition."""
            result = []
            for _, row in partition.iterrows():
                edge = (row[source_col], row[target_col])
                if edge_attrs:
                    attrs = {attr: row[attr] for attr in edge_attrs}
                    result.append((edge[0], edge[1], attrs))
                else:
                    result.append(edge)
            return pd.Series(result)
        
        # Process partitions in parallel
        edge_series = ddf[required_cols].map_partitions(
            extract_edges,
            meta=pd.Series([], dtype=object)
        )
        
        # Compute and collect edges
        edges = edge_series.compute()
        
        # Build graph
        graph_class = nx.DiGraph if directed else nx.Graph
        G = graph_class()
        
        if edge_attrs:
            G.add_edges_from(edges)
        else:
            G.add_edges_from(edges)
        
        logger.info(
            f"Built graph: {G.number_of_nodes():,} nodes, "
            f"{G.number_of_edges():,} edges"
        )
        
        return G
    
    def compute_node_features(
        self,
        G: nx.Graph,
        feature_funcs: Dict[str, Any],
        partition_nodes: int = 10000
    ) -> pd.DataFrame:
        """Compute node features in parallel.
        
        Args:
            G: NetworkX graph
            feature_funcs: Dict of feature_name -> computation function
            partition_nodes: Nodes per partition
            
        Returns:
            DataFrame with node features
        """
        if not DASK_AVAILABLE or self.client is None:
            raise RuntimeError("Dask client not initialized. Use context manager.")
        
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        logger.info(f"Computing {len(feature_funcs)} features for {n_nodes:,} nodes")
        
        # Partition nodes
        node_partitions = [
            nodes[i:i + partition_nodes]
            for i in range(0, n_nodes, partition_nodes)
        ]
        
        def compute_partition_features(node_list):
            """Compute features for a partition of nodes."""
            features = {}
            for node in node_list:
                node_features = {}
                for feat_name, feat_func in feature_funcs.items():
                    try:
                        node_features[feat_name] = feat_func(G, node)
                    except Exception as e:
                        logger.warning(f"Error computing {feat_name} for {node}: {e}")
                        node_features[feat_name] = None
                features[node] = node_features
            return pd.DataFrame.from_dict(features, orient="index")
        
        # Compute features in parallel
        delayed_results = [
            dask.delayed(compute_partition_features)(partition)
            for partition in node_partitions
        ]
        
        results = dask.compute(*delayed_results)
        
        # Combine results
        features_df = pd.concat(results, axis=0)
        
        logger.info(f"Computed features: {features_df.shape}")
        
        return features_df


def should_use_dask(n_edges: int, threshold: int = 1_000_000) -> bool:
    """Determine if Dask should be used based on graph size.
    
    Args:
        n_edges: Number of edges in graph
        threshold: Edge count threshold for using Dask
        
    Returns:
        True if Dask should be used, False otherwise
    """
    if not DASK_AVAILABLE:
        return False
    
    return n_edges >= threshold


def build_graph_auto(
    edges_df: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    edge_attrs: Optional[List[str]] = None,
    directed: bool = True,
    backend: str = "auto",
    dask_threshold: int = 1_000_000,
    **dask_kwargs
) -> nx.Graph:
    """Build graph with automatic backend selection.
    
    Automatically chooses between NetworkX (single-threaded) and Dask
    (parallel) based on graph size.
    
    Args:
        edges_df: DataFrame with edge data
        source_col: Source node column
        target_col: Target node column
        edge_attrs: Additional edge attribute columns
        directed: Whether to build directed graph
        backend: "auto", "networkx", or "dask"
        dask_threshold: Edge count threshold for Dask
        **dask_kwargs: Additional arguments for DaskGraphBuilder
        
    Returns:
        NetworkX graph
    """
    n_edges = len(edges_df)
    
    # Determine backend
    if backend == "auto":
        use_dask = should_use_dask(n_edges, dask_threshold)
        selected_backend = "dask" if use_dask else "networkx"
    else:
        selected_backend = backend
    
    logger.info(
        f"Building graph with {n_edges:,} edges using {selected_backend} backend"
    )
    
    # Build with selected backend
    if selected_backend == "dask":
        if not DASK_AVAILABLE:
            logger.warning("Dask not available, falling back to NetworkX")
            selected_backend = "networkx"
        else:
            with DaskGraphBuilder(**dask_kwargs) as builder:
                return builder.build_graph(
                    edges_df, source_col, target_col, edge_attrs, directed
                )
    
    # NetworkX fallback
    graph_class = nx.DiGraph if directed else nx.Graph
    G = graph_class()
    
    edge_attrs = edge_attrs or []
    for _, row in edges_df.iterrows():
        edge = (row[source_col], row[target_col])
        if edge_attrs:
            attrs = {attr: row[attr] for attr in edge_attrs}
            G.add_edge(edge[0], edge[1], **attrs)
        else:
            G.add_edge(edge[0], edge[1])
    
    logger.info(
        f"Built graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )
    
    return G
