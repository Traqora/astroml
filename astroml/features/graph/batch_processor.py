"""Batch processing for large-scale graph operations.

This module provides optimized batch processing for graph building and
feature computation, using itertools.islice for memory-efficient iteration
and tqdm for progress tracking.

Key features:
- Configurable batch size for edge processing
- Memory-efficient iteration with itertools.islice
- Progress tracking with tqdm
- Sparse matrix optimization for adjacency construction
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# Default batch size for processing edges (10,000 edges per batch)
DEFAULT_BATCH_SIZE = 10_000

# Recommended batch sizes based on graph size
BATCH_SIZE_RECOMMENDATIONS = {
    "small": 5_000,      # < 10k edges
    "medium": 10_000,    # 10k - 100k edges
    "large": 25_000,     # 100k - 1M edges
    "xlarge": 50_000,    # > 1M edges
}


def get_recommended_batch_size(n_edges: int) -> int:
    """Get recommended batch size based on graph size.
    
    Args:
        n_edges: Number of edges in the graph
        
    Returns:
        Recommended batch size for processing
    """
    if n_edges < 10_000:
        return BATCH_SIZE_RECOMMENDATIONS["small"]
    elif n_edges < 100_000:
        return BATCH_SIZE_RECOMMENDATIONS["medium"]
    elif n_edges < 1_000_000:
        return BATCH_SIZE_RECOMMENDATIONS["large"]
    else:
        return BATCH_SIZE_RECOMMENDATIONS["xlarge"]


def batch_edges(
    edges: Iterable[Dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Generator[List[Dict[str, Any]], None, None]:
    """Batch edges using itertools.islice for memory-efficient iteration.
    
    This function uses itertools.islice to process edges in batches without
    loading the entire edge list into memory at once.
    
    Args:
        edges: Iterable of edge dictionaries
        batch_size: Number of edges per batch
        
    Yields:
        Lists of edge dictionaries, each of size <= batch_size
    """
    iterator = iter(edges)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def build_adjacency_matrix_sparse(
    edges: Iterable[Dict[str, Any]],
    node_index: Optional[Dict[str, int]] = None,
    directed: bool = True,
) -> Tuple[csr_matrix, Dict[str, int]]:
    """Build sparse adjacency matrix from edges.
    
    Uses scipy.sparse.csr_matrix for memory-efficient storage of large graphs.
    
    Args:
        edges: Iterable of edge dictionaries with 'src' and 'dst' keys
        node_index: Optional mapping from node IDs to matrix indices.
                    If None, built from edges.
        directed: Whether the graph is directed
        
    Returns:
        Tuple of (sparse adjacency matrix, node index mapping)
    """
    # Build node index if not provided
    if node_index is None:
        nodes: Set[str] = set()
        for edge in edges:
            src = edge.get("src")
            dst = edge.get("dst")
            if src:
                nodes.add(src)
            if dst:
                nodes.add(dst)
        node_list = sorted(nodes)
        node_index = {node: i for i, node in enumerate(node_list)}
    
    n_nodes = len(node_index)
    if n_nodes == 0:
        return csr_matrix((0, 0)), node_index
    
    # Collect edge data for sparse matrix construction
    row_indices = []
    col_indices = []
    data = []
    
    for edge in edges:
        src = edge.get("src")
        dst = edge.get("dst")
        weight = edge.get("amount", 1.0)
        
        if src in node_index and dst in node_index:
            row_indices.append(node_index[src])
            col_indices.append(node_index[dst])
            data.append(weight)
            
            if not directed:
                # Add reverse edge for undirected graphs
                row_indices.append(node_index[dst])
                col_indices.append(node_index[src])
                data.append(weight)
    
    # Create sparse adjacency matrix
    adj_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_nodes, n_nodes),
        dtype=np.float64
    )
    
    return adj_matrix, node_index


class BatchGraphProcessor:
    """Batch processor for large-scale graph operations.
    
    This class provides methods to process graph operations in batches
    with progress tracking and memory-efficient iteration.
    """
    
    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
        progress_desc: str = "Processing edges"
    ):
        """Initialize batch graph processor.
        
        Args:
            batch_size: Number of edges per batch
            show_progress: Whether to show progress bar with tqdm
            progress_desc: Description for progress bar
        """
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.progress_desc = progress_desc
    
    def process_edges(
        self,
        edges: Iterable[Dict[str, Any]],
        process_fn: callable,
        **kwargs
    ) -> List[Any]:
        """Process edges in batches with optional progress tracking.
        
        Args:
            edges: Iterable of edge dictionaries
            process_fn: Function to apply to each batch of edges.
                        Should accept a list of edges and return a result.
            **kwargs: Additional arguments to pass to process_fn
            
        Returns:
            List of results from each batch
        """
        results = []
        
        if self.show_progress:
            try:
                from tqdm import tqdm
                
                # Count total edges for progress bar
                if hasattr(edges, "__len__"):
                    total = len(edges)
                    edge_iterator = edges
                else:
                    # Materialize to count (may be expensive for large iterables)
                    edges_list = list(edges)
                    total = len(edges_list)
                    edge_iterator = edges_list
                
                for batch in tqdm(
                    batch_edges(edge_iterator, self.batch_size),
                    total=(total + self.batch_size - 1) // self.batch_size,
                    desc=self.progress_desc
                ):
                    result = process_fn(batch, **kwargs)
                    results.append(result)
            except ImportError:
                logger.warning("tqdm not installed, falling back to no progress bar")
                self.show_progress = False
                for batch in batch_edges(edges, self.batch_size):
                    result = process_fn(batch, **kwargs)
                    results.append(result)
        else:
            for batch in batch_edges(edges, self.batch_size):
                result = process_fn(batch, **kwargs)
                results.append(result)
        
        return results
    
    def build_graph_incremental(
        self,
        edges: Iterable[Dict[str, Any]],
        graph_builder: callable,
        **kwargs
    ) -> Any:
        """Build graph incrementally from edge batches.
        
        Args:
            edges: Iterable of edge dictionaries
            graph_builder: Function that takes edges and returns a graph object.
                          Should support incremental updates.
            **kwargs: Additional arguments for graph_builder
            
        Returns:
            Built graph object
        """
        graph = None
        
        if self.show_progress:
            try:
                from tqdm import tqdm
                
                if hasattr(edges, "__len__"):
                    total = len(edges)
                    edge_iterator = edges
                else:
                    edges_list = list(edges)
                    total = len(edges_list)
                    edge_iterator = edges_list
                
                for batch in tqdm(
                    batch_edges(edge_iterator, self.batch_size),
                    total=(total + self.batch_size - 1) // self.batch_size,
                    desc=self.progress_desc
                ):
                    if graph is None:
                        graph = graph_builder(batch, **kwargs)
                    else:
                        # Incremental update
                        graph = graph_builder(batch, graph=graph, **kwargs)
            except ImportError:
                logger.warning("tqdm not installed, falling back to no progress bar")
                for batch in batch_edges(edges, self.batch_size):
                    if graph is None:
                        graph = graph_builder(batch, **kwargs)
                    else:
                        graph = graph_builder(batch, graph=graph, **kwargs)
        else:
            for batch in batch_edges(edges, self.batch_size):
                if graph is None:
                    graph = graph_builder(batch, **kwargs)
                else:
                    graph = graph_builder(batch, graph=graph, **kwargs)
        
        return graph


def compute_degree_centrality_batched(
    edges: Iterable[Dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
    weighted: bool = False
) -> pd.Series:
    """Compute degree centrality using batch processing.
    
    Args:
        edges: Iterable of edge dictionaries
        batch_size: Number of edges per batch
        show_progress: Whether to show progress bar
        weighted: Whether to use weighted degree
        
    Returns:
        pandas.Series with degree centrality values
    """
    from collections import defaultdict
    
    degree_counts = defaultdict(float)
    processor = BatchGraphProcessor(
        batch_size=batch_size,
        show_progress=show_progress,
        progress_desc="Computing degree centrality"
    )
    
    def process_batch(batch: List[Dict[str, Any]]) -> Dict[str, float]:
        batch_counts = defaultdict(float)
        for edge in batch:
            src = edge.get("src")
            dst = edge.get("dst")
            weight = float(edge.get("amount", 1.0) or 1.0) if weighted else 1.0
            
            if src:
                batch_counts[src] += weight
            if dst:
                batch_counts[dst] += weight
        return batch_counts
    
    batch_results = processor.process_edges(edges, process_batch)
    
    # Aggregate results
    for batch_counts in batch_results:
        for node, count in batch_counts.items():
            degree_counts[node] += count
    
    # Normalize
    node_set = set(degree_counts.keys())
    n_nodes = len(node_set)
    if n_nodes <= 1:
        return pd.Series(0.0, index=list(node_set), dtype=float)
    
    centrality = {node: count / (n_nodes - 1) for node, count in degree_counts.items()}
    
    return pd.Series(centrality, dtype=float).sort_index()
