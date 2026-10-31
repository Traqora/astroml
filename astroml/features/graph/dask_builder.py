"""Dask-based parallel graph building for large-scale graphs.

This module provides distributed graph building using Dask for graphs
with 1M+ edges where parallel processing provides significant speedup.

A graph builder is only useful if it can bound the memory it uses. Passing
``memory_limit`` to :class:`~dask.distributed.LocalCluster` caps each
*worker*, but the previous implementation then pulled every edge back onto
the driver with a single ``compute()`` call, so a graph that fit across the
cluster could still OOM the process that asked for it. :func:`parse_memory_limit`
and :func:`enforce_memory_cap` make that ceiling explicit and testable, and
:meth:`DaskGraphBuilder.build_graph` now consumes one partition at a time so
the driver's peak footprint is a partition, not the whole edge set.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import networkx as nx
import pandas as pd

from astroml.utils.exceptions import AstroMLError

logger = logging.getLogger(__name__)

#: Rough per-edge cost of a Python ``(src, dst[, attrs])`` tuple held in
#: memory. Deliberately generous so the cap errs on the safe side.
DEFAULT_EDGE_BYTES = 256

#: Extra bytes charged per edge-attribute key/value pair.
ATTRIBUTE_BYTES = 64

_MEMORY_UNITS: dict[str, int] = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
}
_MEMORY_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]*)\s*$")


def parse_memory_limit(limit: str | int) -> int:
    """Parse a human-readable memory limit into bytes.

    Args:
        limit: Either a plain integer number of bytes, or a string such as
            ``"4GB"``, ``"512MB"``, ``"1024KB"`` or ``"2048"``. Units are
            case-insensitive and an optional ``i`` (as in ``"4GiB"``) is
            accepted.

    Returns:
        The limit in bytes.

    Raises:
        ValueError: If the string is not a non-negative number followed by a
            recognised unit.
    """
    if isinstance(limit, bool) or not isinstance(limit, (int, str)):
        raise ValueError(f"memory limit must be an int or str, got {type(limit).__name__}")
    if isinstance(limit, int):
        if limit < 0:
            raise ValueError(f"memory limit must be non-negative, got {limit}")
        return limit

    match = _MEMORY_RE.match(limit)
    if match is None:
        raise ValueError(f"cannot parse memory limit {limit!r}; expected e.g. '4GB'")
    value = float(match.group("value"))
    unit = match.group("unit").upper().replace("I", "")
    if unit not in _MEMORY_UNITS:
        raise ValueError(f"unknown memory unit {match.group('unit')!r} in {limit!r}")
    return int(value * _MEMORY_UNITS[unit])


def estimate_edge_memory(
    n_edges: int,
    n_attrs: int = 0,
    *,
    overhead_bytes_per_edge: int = DEFAULT_EDGE_BYTES,
) -> int:
    """Estimate the driver memory a graph of ``n_edges`` edges will occupy.

    Args:
        n_edges: Number of edges that will be materialised.
        n_attrs: Number of attribute values carried by each edge.
        overhead_bytes_per_edge: Baseline per-edge cost before attributes.

    Returns:
        Estimated footprint in bytes.

    Raises:
        ValueError: If ``n_edges`` or ``n_attrs`` is negative.
    """
    if n_edges < 0:
        raise ValueError(f"n_edges must be non-negative, got {n_edges}")
    if n_attrs < 0:
        raise ValueError(f"n_attrs must be non-negative, got {n_attrs}")
    return n_edges * (overhead_bytes_per_edge + ATTRIBUTE_BYTES * n_attrs)


def enforce_memory_cap(
    n_edges: int,
    memory_limit_bytes: int,
    n_attrs: int = 0,
    *,
    overhead_bytes_per_edge: int = DEFAULT_EDGE_BYTES,
) -> int:
    """Return the estimated edge footprint, raising if it exceeds the cap.

    Args:
        n_edges: Number of edges about to be materialised.
        memory_limit_bytes: Maximum estimated driver footprint allowed.
        n_attrs: Number of attribute values carried by each edge.
        overhead_bytes_per_edge: Baseline per-edge cost before attributes.

    Returns:
        The estimated footprint in bytes when it fits within the cap.

    Raises:
        MemoryError: If the estimate exceeds ``memory_limit_bytes``.
        ValueError: If either size argument is negative.
    """
    if memory_limit_bytes < 0:
        raise ValueError(f"memory_limit_bytes must be non-negative, got {memory_limit_bytes}")
    estimated = estimate_edge_memory(
        n_edges, n_attrs, overhead_bytes_per_edge=overhead_bytes_per_edge
    )
    if estimated > memory_limit_bytes:
        raise MemoryError(
            f"building a graph with {n_edges:,} edges would need an estimated "
            f"{estimated / 1024**2:.1f} MiB, above the configured cap of "
            f"{memory_limit_bytes / 1024**2:.1f} MiB; raise the memory cap or "
            "build the graph in smaller pieces"
        )
    return estimated


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
        n_workers: int | None = None,
        threads_per_worker: int = 2,
        memory_limit: str = "4GB",
        max_edges: int | None = None,
    ):
        """Initialize Dask graph builder.

        Args:
            n_workers: Number of Dask workers (default: CPU count)
            threads_per_worker: Threads per worker
            memory_limit: Memory limit per worker, e.g. ``"4GB"``. Also used
                as the driver-side cap on the estimated edge footprint (see
                :meth:`build_graph`).
            max_edges: Optional hard ceiling on the number of edges a single
                :meth:`build_graph` call will accept, independent of the
                byte estimate.

        Raises:
            ImportError: If Dask is not installed.
            ValueError: If ``max_edges`` is negative or ``memory_limit``
                cannot be parsed.
        """
        if max_edges is not None and max_edges < 0:
            raise ValueError(f"max_edges must be non-negative, got {max_edges}")
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is required for DaskGraphBuilder. " "Install with: pip install astroml[dask]"
            )

        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.memory_limit_bytes = parse_memory_limit(memory_limit)
        self.max_edges = max_edges
        self.client: Client | None = None

    def __enter__(self) -> Any:
        """Start Dask cluster."""
        cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
            silence_logs=logging.WARNING,
        )
        self.client = Client(cluster)
        logger.info(f"Started Dask cluster: {self.client.dashboard_link}")
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> Any:
        """Stop Dask cluster."""
        if self.client:
            self.client.close()
            self.client = None

    def build_graph(
        self,
        edges_df: pd.DataFrame,
        source_col: str = "source",
        target_col: str = "target",
        edge_attrs: list[str] | None = None,
        directed: bool = True,
        partition_size: int = 100000,
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

        Raises:
            RuntimeError: If the Dask client has not been started.
            MemoryError: If ``max_edges`` is exceeded, or if the estimated
                driver footprint of the edge set exceeds the configured
                ``memory_limit``.
        """
        if not DASK_AVAILABLE or self.client is None:
            raise RuntimeError("Dask client not initialized. Use context manager.")

        n_edges = len(edges_df)
        edge_attrs = edge_attrs or []

        if self.max_edges is not None and n_edges > self.max_edges:
            raise MemoryError(
                f"refusing to build a graph with {n_edges:,} edges: exceeds "
                f"max_edges={self.max_edges:,}"
            )
        enforce_memory_cap(
            n_edges,
            self.memory_limit_bytes,
            n_attrs=len(edge_attrs),
        )

        logger.info(f"Building graph with {n_edges:,} edges using Dask")

        # Convert to Dask DataFrame
        ddf = dd.from_pandas(edges_df, npartitions=max(1, n_edges // partition_size))

        required_cols = [source_col, target_col] + edge_attrs

        # Extract edge tuples
        def extract_edges(partition) -> Any:
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

        # Build the graph one partition at a time. Computing the whole
        # ``edge_series`` up front would materialise every edge on the
        # driver, defeating the memory cap that was just checked; adding
        # per-partition keeps peak driver memory proportional to a single
        # partition instead of the whole edge set.
        graph_class = nx.DiGraph if directed else nx.Graph
        G = graph_class()
        for partition in ddf[required_cols].to_delayed():
            edges = extract_edges(partition.compute())
            G.add_edges_from(edges)

        logger.info(
            f"Built graph: {G.number_of_nodes():,} nodes, " f"{G.number_of_edges():,} edges"
        )

        return G

    def compute_node_features(
        self, G: nx.Graph, feature_funcs: dict[str, Any], partition_nodes: int = 10000
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
            nodes[i : i + partition_nodes] for i in range(0, n_nodes, partition_nodes)
        ]

        def compute_partition_features(node_list) -> Any:
            """Compute features for a partition of nodes."""
            features = {}
            for node in node_list:
                node_features = {}
                for feat_name, feat_func in feature_funcs.items():
                    try:
                        node_features[feat_name] = feat_func(G, node)
                    except AstroMLError as e:
                        logger.warning(f"Error computing {feat_name} for {node}: {e}")
                        node_features[feat_name] = None
                features[node] = node_features
            return pd.DataFrame.from_dict(features, orient="index")

        # Compute features in parallel
        delayed_results = [
            dask.delayed(compute_partition_features)(partition) for partition in node_partitions
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
    edge_attrs: list[str] | None = None,
    directed: bool = True,
    backend: str = "auto",
    dask_threshold: int = 1_000_000,
    **dask_kwargs,
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

    logger.info(f"Building graph with {n_edges:,} edges using {selected_backend} backend")

    # Build with selected backend
    if selected_backend == "dask":
        if not DASK_AVAILABLE:
            logger.warning("Dask not available, falling back to NetworkX")
            selected_backend = "networkx"
        else:
            with DaskGraphBuilder(**dask_kwargs) as builder:
                return builder.build_graph(edges_df, source_col, target_col, edge_attrs, directed)

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

    logger.info(f"Built graph: {G.number_of_nodes():,} nodes, " f"{G.number_of_edges():,} edges")

    return G
