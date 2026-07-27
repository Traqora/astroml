"""Structural importance metrics for account nodes.

This module provides various centrality and importance measures for nodes
in the transaction graph, helping identify structurally significant accounts.

Features computed:
- degree_centrality: Normalized degree (in + out)
- betweenness_centrality: Betweenness centrality measure
- closeness_centrality: Closeness centrality measure  
- pagerank: PageRank score
- eigenvector_centrality: Eigenvector centrality measure
- clustering_coefficient: Local clustering coefficient
- strength_centrality: Weighted degree centrality (total transaction volume)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import warnings
import itertools

Edge = Dict[str, object]


def compute_degree_centrality(
    edges: Iterable[Edge], 
    nodes: Optional[List[str]] = None,
    weighted: bool = False
) -> pd.Series:
    """Compute degree centrality for nodes.
    
    Args:
        edges: Iterable of edge dictionaries with 'src', 'dst', and optionally 'amount'
        nodes: Optional list of nodes to include. If None, inferred from edges.
        weighted: If True, use weighted degree (strength) based on transaction amounts.
        
    Returns:
        pandas.Series indexed by node with degree centrality values
    """
    degree_counts = defaultdict(float)
    node_set = set() if nodes is None else set(nodes)
    
    for edge in edges:
        src = edge.get('src')
        dst = edge.get('dst')
        weight = float(edge.get('amount', 1.0) or 1.0) if weighted else 1.0
        
        if src is not None:
            degree_counts[src] += weight
            node_set.add(src)
        if dst is not None:
            degree_counts[dst] += weight
            node_set.add(dst)
    
    # Convert to normalized centrality (divide by n-1)
    n_nodes = len(node_set)
    if n_nodes <= 1:
        return pd.Series(0.0, index=list(node_set), dtype=float)
    
    centrality = {node: count / (n_nodes - 1) for node, count in degree_counts.items()}
    
    # Ensure all nodes are present
    for node in node_set:
        if node not in centrality:
            centrality[node] = 0.0
    
    return pd.Series(centrality, dtype=float).sort_index()


def compute_betweenness_centrality(
    edges: Iterable[Edge],
    nodes: Optional[List[str]] = None,
    normalized: bool = True,
    sample_size: Optional[int] = None
) -> pd.Series:
    """Compute betweenness centrality using Brandes' algorithm.
    
    Args:
        edges: Iterable of edge dictionaries with 'src', 'dst'
        nodes: Optional list of nodes to include. If None, inferred from edges.
        normalized: Whether to normalize the scores.
        sample_size: Optional sample size for approximation on large graphs.
        
    Returns:
        pandas.Series indexed by node with betweenness centrality values
    """
    # Build adjacency list
    adj = defaultdict(set)
    node_set = set() if nodes is None else set(nodes)
    
    for edge in edges:
        src = edge.get('src')
        dst = edge.get('dst')
        if src is not None and dst is not None:
            adj[src].add(dst)
            adj[dst].add(src)  # Undirected for betweenness
            node_set.update([src, dst])
    
    nodes_list = list(node_set)
    
    # Sample nodes for large graphs if requested
    if sample_size is not None and len(nodes_list) > sample_size:
        np.random.seed(42)  # For reproducibility
        nodes_list = np.random.choice(nodes_list, sample_size, replace=False).tolist()
    
    betweenness = defaultdict(float)
    
    for s in nodes_list:
        # Single-source shortest paths from s
        S = []
        P = defaultdict(list)
        sigma = defaultdict(float)
        sigma[s] = 1.0
        dist = defaultdict(lambda: -1)
        dist[s] = 0
        
        Q = deque([s])
        
        while Q:
            v = Q.popleft()
            S.append(v)
            
            for w in adj[v]:
                if dist[w] < 0:
                    Q.append(w)
                    dist[w] = dist[v] + 1
                
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        
        # Accumulation
        delta = defaultdict(float)
        
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            
            if w != s:
                betweenness[w] += delta[w]
    
    # Normalize
    if normalized and len(node_set) > 2:
        scale = 2.0 / ((len(node_set) - 1) * (len(node_set) - 2))
        for node in betweenness:
            betweenness[node] *= scale
    
    # Ensure all nodes are present
    for node in node_set:
        if node not in betweenness:
            betweenness[node] = 0.0
    
    return pd.Series(betweenness, dtype=float).sort_index()


def compute_closeness_centrality(
    edges: Iterable[Edge],
    nodes: Optional[List[str]] = None,
    disconnected: bool = False
) -> pd.Series:
    """Compute closeness centrality for nodes.
    
    Args:
        edges: Iterable of edge dictionaries with 'src', 'dst'
        nodes: Optional list of nodes to include. If None, inferred from edges.
        disconnected: Whether to handle disconnected components.
        
    Returns:
        pandas.Series indexed by node with closeness centrality values
    """
    # Build adjacency list
    adj = defaultdict(set)
    node_set = set() if nodes is None else set(nodes)
    
    for edge in edges:
        src = edge.get('src')
        dst = edge.get('dst')
        if src is not None and dst is not None:
            adj[src].add(dst)
            adj[dst].add(src)  # Undirected for closeness
            node_set.update([src, dst])
    
    closeness = {}
    nodes_list = list(node_set)
    
    for s in nodes_list:
        # BFS from s
        dist = {s: 0}
        Q = deque([s])
        visited = {s}
        
        while Q:
            v = Q.popleft()
            for w in adj[v]:
                if w not in visited:
                    visited.add(w)
                    dist[w] = dist[v] + 1
                    Q.append(w)
        
        if len(visited) <= 1:
            closeness[s] = 0.0
        elif disconnected:
            # Handle disconnected components
            total_dist = sum(dist.values())
            n_reachable = len(visited)
            closeness[s] = (n_reachable - 1) / total_dist if total_dist > 0 else 0.0
        else:
            # Only compute if fully connected
            if len(visited) == len(node_set):
                total_dist = sum(dist.values())
                closeness[s] = (len(node_set) - 1) / total_dist if total_dist > 0 else 0.0
            else:
                closeness[s] = 0.0
    
    return pd.Series(closeness, dtype=float).sort_index()


def compute_pagerank(
    edges: Iterable[Edge],
    nodes: Optional[List[str]] = None,
    alpha: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    weighted: bool = True
) -> pd.Series:
    """Compute PageRank scores for nodes.
    
    Args:
        edges: Iterable of edge dictionaries with 'src', 'dst', and optionally 'amount'
        nodes: Optional list of nodes to include. If None, inferred from edges.
        alpha: Damping factor (typically 0.85).
        max_iter: Maximum number of iterations.
        tolerance: Convergence tolerance.
        weighted: If True, use edge weights for transition probabilities.
        
    Returns:
        pandas.Series indexed by node with PageRank scores
    """
    # Build adjacency and out-degree information
    adj = defaultdict(list)
    out_degree = defaultdict(float)
    node_set = set() if nodes is None else set(nodes)
    
    for edge in edges:
        src = edge.get('src')
        dst = edge.get('dst')
        weight = float(edge.get('amount', 1.0) or 1.0) if weighted else 1.0
        
        if src is not None and dst is not None:
            adj[src].append((dst, weight))
            out_degree[src] += weight
            node_set.update([src, dst])
    
    nodes_list = list(node_set)
    n_nodes = len(nodes_list)
    
    if n_nodes == 0:
        return pd.Series(dtype=float)
    
    # Initialize PageRank
    pagerank = {node: 1.0 / n_nodes for node in nodes_list}
    
    # Power iteration
    for iteration in range(max_iter):
        new_pagerank = {}
        dangling_sum = 0.0
        
        # Handle dangling nodes (no outgoing edges)
        for node in nodes_list:
            if out_degree[node] == 0:
                dangling_sum += pagerank[node]
        
        # Compute new PageRank
        for node in nodes_list:
            rank = (1.0 - alpha) / n_nodes  # Random jump
            rank += alpha * dangling_sum / n_nodes  # Dangling contribution
            
            # Contribution from incoming edges
            for src, (dst, weight) in [(s, (d, w)) for s in adj for d, w in adj[s] if d == node]:
                if out_degree[src] > 0:
                    rank += alpha * pagerank[src] * (weight / out_degree[src])
            
            new_pagerank[node] = rank
        
        # Check convergence
        diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in nodes_list)
        if diff < tolerance:
            break
        
        pagerank = new_pagerank
    
    return pd.Series(pagerank, dtype=float).sort_index()


def compute_clustering_coefficient(
    edges: Iterable[Edge],
    nodes: Optional[List[str]] = None
) -> pd.Series:
    """Compute local clustering coefficient for nodes.
    
    Args:
        edges: Iterable of edge dictionaries with 'src', 'dst'
        nodes: Optional list of nodes to include. If None, inferred from edges.
        
    Returns:
        pandas.Series indexed by node with clustering coefficient values
    """
    # Build adjacency list
    adj = defaultdict(set)
    node_set = set() if nodes is None else set(nodes)
    
    for edge in edges:
        src = edge.get('src')
        dst = edge.get('dst')
        if src is not None and dst is not None:
            adj[src].add(dst)
            adj[dst].add(src)  # Undirected for clustering
            node_set.update([src, dst])
    
    clustering = {}
    
    for node in node_set:
        neighbors = adj[node]
        k = len(neighbors)
        
        if k < 2:
            clustering[node] = 0.0
        else:
            # Count edges between neighbors
            neighbor_edges = 0
            for u in neighbors:
                for v in neighbors:
                    if u < v and v in adj[u]:  # Count each edge once
                        neighbor_edges += 1
            
            # Clustering coefficient = 2 * edges / (k * (k-1))
            clustering[node] = (2.0 * neighbor_edges) / (k * (k - 1))
    
    return pd.Series(clustering, dtype=float).sort_index()


def compute_strength_centrality(
    edges: Iterable[Edge],
    nodes: Optional[List[str]] = None,
    direction: str = 'both'
) -> pd.Series:
    """Compute strength centrality (weighted degree) based on transaction amounts.
    
    Args:
        edges: Iterable of edge dictionaries with 'src', 'dst', 'amount'
        nodes: Optional list of nodes to include. If None, inferred from edges.
        direction: 'in', 'out', or 'both' for in/out/total strength.
        
    Returns:
        pandas.Series indexed by node with strength centrality values
    """
    strength = defaultdict(float)
    node_set = set() if nodes is None else set(nodes)
    
    for edge in edges:
        src = edge.get('src')
        dst = edge.get('dst')
        amount = float(edge.get('amount', 0.0) or 0.0)
        
        if src is not None:
            if direction in ['out', 'both']:
                strength[src] += amount
            node_set.add(src)
        
        if dst is not None:
            if direction in ['in', 'both']:
                strength[dst] += amount
            node_set.add(dst)
    
    # Ensure all nodes are present
    for node in node_set:
        if node not in strength:
            strength[node] = 0.0
    
    return pd.Series(strength, dtype=float).sort_index()


def compute_structural_importance_metrics(
    edges: Iterable[Edge],
    nodes: Optional[List[str]] = None,
    include_betweenness: bool = True,
    include_closeness: bool = True,
    include_eigenvector: bool = False,
    pagerank_sample_size: Optional[int] = None,
    betweenness_sample_size: Optional[int] = None
) -> pd.DataFrame:
    """Compute comprehensive structural importance metrics for account nodes.
    
    This is the main pipeline step function that calculates various centrality
    and importance measures for nodes in the transaction graph.
    
    Args:
        edges: Iterable of edge dictionaries with 'src', 'dst', and optionally 'amount'
        nodes: Optional list of nodes to include. If None, inferred from edges.
        include_betweenness: Whether to compute betweenness centrality (expensive).
        include_closeness: Whether to compute closeness centrality.
        include_eigenvector: Whether to compute eigenvector centrality (requires scipy).
        pagerank_sample_size: Optional sample size for PageRank approximation.
        betweenness_sample_size: Optional sample size for betweenness approximation.
        
    Returns:
        pandas.DataFrame indexed by node with structural importance metrics:
        - degree_centrality: Normalized degree centrality
        - strength_centrality: Weighted degree based on transaction volume
        - pagerank: PageRank score
        - clustering_coefficient: Local clustering coefficient
        - betweenness_centrality: Betweenness centrality (if computed)
        - closeness_centrality: Closeness centrality (if computed)
        - eigenvector_centrality: Eigenvector centrality (if computed)
    """
    metrics = {}
    
    # Basic centrality measures
    metrics['degree_centrality'] = compute_degree_centrality(edges, nodes, weighted=False)
    metrics['strength_centrality'] = compute_strength_centrality(edges, nodes, direction='both')
    metrics['pagerank'] = compute_pagerank(edges, nodes, sample_size=pagerank_sample_size)
    metrics['clustering_coefficient'] = compute_clustering_coefficient(edges, nodes)
    
    # More expensive measures (optional)
    if include_betweenness:
        try:
            metrics['betweenness_centrality'] = compute_betweenness_centrality(
                edges, nodes, sample_size=betweenness_sample_size
            )
        except Exception as e:
            warnings.warn(f"Failed to compute betweenness centrality: {e}")
            metrics['betweenness_centrality'] = pd.Series(0.0, index=metrics['degree_centrality'].index)
    
    if include_closeness:
        try:
            metrics['closeness_centrality'] = compute_closeness_centrality(edges, nodes)
        except Exception as e:
            warnings.warn(f"Failed to compute closeness centrality: {e}")
            metrics['closeness_centrality'] = pd.Series(0.0, index=metrics['degree_centrality'].index)
    
    if include_eigenvector:
        try:
            metrics['eigenvector_centrality'] = compute_eigenvector_centrality(edges, nodes)
        except Exception as e:
            warnings.warn(f"Failed to compute eigenvector centrality: {e}")
            metrics['eigenvector_centrality'] = pd.Series(0.0, index=metrics['degree_centrality'].index)
    
    # Combine into DataFrame
    df = pd.DataFrame(metrics)
    
    # Fill any missing values with 0
    df = df.fillna(0.0)
    
    return df.sort_index()


def compute_eigenvector_centrality(
    edges: Iterable[Edge],
    nodes: Optional[List[str]] = None,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    weighted: bool = True
) -> pd.Series:
    """Compute eigenvector centrality for nodes.
    
    Args:
        edges: Iterable of edge dictionaries with 'src', 'dst', and optionally 'amount'
        nodes: Optional list of nodes to include. If None, inferred from edges.
        max_iter: Maximum number of iterations.
        tolerance: Convergence tolerance.
        weighted: If True, use edge weights.
        
    Returns:
        pandas.Series indexed by node with eigenvector centrality values
    """
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError:
        warnings.warn("SciPy is required for eigenvector centrality. Install with: pip install scipy")
        return pd.Series(dtype=float)
    
    # Build adjacency matrix
    node_set = set() if nodes is None else set(nodes)
    node_list = list(node_set)
    node_index = {node: i for i, node in enumerate(node_list)}
    
    # Collect edges and build matrix
    row_ind = []
    col_ind = []
    data = []
    
    for edge in edges:
        src = edge.get('src')
        dst = edge.get('dst')
        weight = float(edge.get('amount', 1.0) or 1.0) if weighted else 1.0
        
        if src is not None and dst is not None:
            node_set.update([src, dst])
            if src in node_index and dst in node_index:
                row_ind.append(node_index[src])
                col_ind.append(node_index[dst])
                data.append(weight)
    
    if len(node_list) == 0:
        return pd.Series(dtype=float)
    
    # Create sparse matrix
    n = len(node_list)
    adj_matrix = sp.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    
    # Power iteration method
    x = np.ones(n) / np.sqrt(n)
    
    for _ in range(max_iter):
        x_new = adj_matrix.dot(x)
        norm = np.linalg.norm(x_new)
        
        if norm == 0:
            break
            
        x_new = x_new / norm
        
        if np.linalg.norm(x_new - x) < tolerance:
            break
            
        x = x_new
    
    return pd.Series(x, index=node_list, dtype=float).sort_index()
