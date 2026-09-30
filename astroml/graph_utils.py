"""Graph utilities for AstroML - PyTorch Geometric conversion and graph analysis functions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

try:
    from torch_geometric.data import Data

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = Any  # type: ignore[misc,assignment]

try:
    from torch_geometric.utils import k_hop_subgraph
except ImportError:
    k_hop_subgraph = None


def graph_to_pyg_data(
    node_features: np.ndarray | list[list[float]],
    edge_index: np.ndarray | list[list[int]],
    edge_features: np.ndarray | list[list[float]] | None = None,
    node_labels: np.ndarray | list[int] | None = None,
) -> Data:
    """Convert graph arrays to a PyTorch Geometric Data object.

    Args:
        node_features: Array of shape [num_nodes, num_node_features]
        edge_index: Array of shape [2, num_edges] or [num_edges, 2]
        edge_features: Optional array of shape [num_edges, num_edge_features]
        node_labels: Optional array of shape [num_nodes]

    Returns:
        torch_geometric.data.Data object
    """
    # Convert inputs to numpy arrays if they are lists
    if isinstance(node_features, list):
        node_features = np.array(node_features, dtype=np.float32)
    elif isinstance(node_features, np.ndarray):
        node_features = node_features.astype(np.float32)

    if isinstance(edge_index, list):
        try:
            edge_index = np.array(edge_index, dtype=np.int64)
        except (ValueError, TypeError):
            # Handle inhomogeneous lists
            edge_index = np.array(edge_index, dtype=object)
            if edge_index.ndim != 2:
                raise ValueError(f"edge_index must be 2D array, got shape {edge_index.shape}")
            edge_index = edge_index.astype(np.int64)
    elif isinstance(edge_index, np.ndarray):
        edge_index = edge_index.astype(np.int64)

    # Validate node features
    if len(node_features.shape) != 2:
        raise ValueError(f"node_features must be 2D array, got shape {node_features.shape}")

    num_nodes = node_features.shape[0]

    # Handle edge index format conversion
    if edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
        # Format is [num_edges, 2], need to transpose to [2, num_edges]
        edge_index = edge_index.T
    elif edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges] or [num_edges, 2], got {edge_index.shape}"
        )

    # Validate edge indices are within bounds (skip if no edges)
    if edge_index.size > 0:
        if edge_index.max() >= num_nodes:
            raise ValueError(
                f"Edge index contains node ID {edge_index.max()} which is >= num_nodes ({num_nodes})"
            )
        if edge_index.min() < 0:
            raise ValueError("Edge index contains negative node IDs")

    # Convert to tensors
    x = torch.from_numpy(node_features)  # [num_nodes, num_node_features]
    edge_index_tensor = torch.from_numpy(edge_index)  # [2, num_edges]

    # Handle edge features
    edge_attr = None
    if edge_features is not None:
        if isinstance(edge_features, list):
            edge_features = np.array(edge_features, dtype=np.float32)
        elif isinstance(edge_features, np.ndarray):
            edge_features = edge_features.astype(np.float32)

        # Validate edge features shape
        if len(edge_features.shape) != 2:
            raise ValueError(f"edge_features must be 2D array, got shape {edge_features.shape}")

        if edge_features.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_features shape mismatch: expected {edge_index.shape[1]} edges, "
                f"got {edge_features.shape[0]}"
            )

        edge_attr = torch.from_numpy(edge_features)  # [num_edges, num_edge_features]

    # Handle node labels
    y = None
    if node_labels is not None:
        if isinstance(node_labels, list):
            node_labels = np.array(node_labels)
        elif isinstance(node_labels, np.ndarray):
            node_labels = node_labels

        # Validate node labels shape
        if len(node_labels.shape) > 1:
            raise ValueError(f"node_labels must be 1D array, got shape {node_labels.shape}")

        if node_labels.shape[0] != num_nodes:
            raise ValueError(
                f"node_labels shape mismatch: expected {num_nodes} nodes, "
                f"got {node_labels.shape[0]}"
            )

        y = torch.from_numpy(node_labels).to(torch.int64)  # [num_nodes]

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr, y=y, num_nodes=num_nodes)

    return data


def build_transaction_pyg_graph(
    transactions: list[dict[str, Any]],
    node_feature_dict: Mapping[str, Sequence[float]] | None = None,
    label_dict: Mapping[str, int] | None = None,
    default_feature_dim: int = 8,
) -> tuple[Data, dict[str, int]]:
    """Construct a PyTorch Geometric Data object from raw transaction records.

    Parameters
    ----------
    transactions : list[dict[str, Any]]
        List of transaction dictionaries. Each dict should have 'source_account'/'from',
        'destination_account'/'to', 'amount', and optional 'timestamp'/'asset'.
    node_feature_dict : Mapping[str, Sequence[float]] | None
        Optional mapping from account ID to feature vector.
    label_dict : Mapping[str, int] | None
        Optional mapping from account ID to integer class label (e.g. 0=legit, 1=fraud).
    default_feature_dim : int
        Default feature dimension if no node_feature_dict is provided.

    Returns
    -------
    tuple[Data, dict[str, int]]
        PyG Data object and mapping from account string ID to node index [0..N-1].
    """
    node_to_id: dict[str, int] = {}
    edges_src: list[int] = []
    edges_dst: list[int] = []
    edge_weights: list[list[float]] = []

    def get_or_add_node(acc: str) -> int:
        if acc not in node_to_id:
            node_to_id[acc] = len(node_to_id)
        return node_to_id[acc]

    for tx in transactions:
        src = str(tx.get("source_account") or tx.get("from") or tx.get("source") or "")
        dst = str(tx.get("destination_account") or tx.get("to") or tx.get("destination") or "")
        if not src or not dst or src == dst:
            continue

        src_id = get_or_add_node(src)
        dst_id = get_or_add_node(dst)

        amount = float(tx.get("amount", 1.0))
        fee = float(tx.get("fee", 0.0))
        edge_type = float(tx.get("edge_type", 0.0))

        edges_src.append(src_id)
        edges_dst.append(dst_id)
        edge_weights.append([amount, fee, edge_type])

    num_nodes = len(node_to_id)
    if num_nodes == 0:
        x = torch.empty((0, default_feature_dim), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=0), node_to_id

    # Node features
    if node_feature_dict:
        feature_list = []
        feat_dim = (
            len(next(iter(node_feature_dict.values())))
            if node_feature_dict
            else default_feature_dim
        )
        for acc in node_to_id:
            if acc in node_feature_dict:
                feature_list.append(list(node_feature_dict[acc]))
            else:
                feature_list.append([0.0] * feat_dim)
        x_tensor = torch.tensor(feature_list, dtype=torch.float32)
    else:
        # In-degree, out-degree, total volume features
        in_degrees = np.zeros(num_nodes, dtype=np.float32)
        out_degrees = np.zeros(num_nodes, dtype=np.float32)
        in_volumes = np.zeros(num_nodes, dtype=np.float32)
        out_volumes = np.zeros(num_nodes, dtype=np.float32)

        for s, d, w in zip(edges_src, edges_dst, edge_weights):
            out_degrees[s] += 1
            in_degrees[d] += 1
            out_volumes[s] += w[0]
            in_volumes[d] += w[0]

        total_degrees = in_degrees + out_degrees
        total_volumes = in_volumes + out_volumes
        ratio_in_out = np.where(
            out_degrees > 0, in_degrees / np.maximum(out_degrees, 1.0), in_degrees
        )
        vol_ratio = np.where(
            out_volumes > 0, in_volumes / np.maximum(out_volumes, 1e-6), in_volumes
        )

        features = np.stack(
            [
                in_degrees,
                out_degrees,
                total_degrees,
                in_volumes,
                out_volumes,
                total_volumes,
                ratio_in_out,
                vol_ratio,
            ],
            axis=1,
        )
        x_tensor = torch.tensor(features, dtype=torch.float32)

    # Edge Index & Edge Attr
    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)

    # Labels
    y_tensor = None
    if label_dict:
        labels = [label_dict.get(acc, 0) for acc in node_to_id]
        y_tensor = torch.tensor(labels, dtype=torch.long)

    pyg_data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_tensor,
        num_nodes=num_nodes,
    )
    return pyg_data, node_to_id


def extract_k_hop_subgraph(
    node_idx: int | list[int],
    num_hops: int,
    edge_index: torch.Tensor,
    num_nodes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the k-hop ego subgraph around node_idx.

    Parameters
    ----------
    node_idx : int | list[int]
        Target node index or indices.
    num_hops : int
        Number of hops / neighborhood distance.
    edge_index : torch.Tensor
        Graph edge index [2, E].
    num_nodes : int | None
        Total number of nodes in full graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        (subset_nodes, sub_edge_index, mapping, edge_mask)
    """
    if _HAS_PYG and k_hop_subgraph is not None:
        target = [node_idx] if isinstance(node_idx, int) else node_idx
        return k_hop_subgraph(
            node_idx=target,
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )

    # Pure PyTorch fallback for k-hop subgraph extraction
    target_set = {node_idx} if isinstance(node_idx, int) else set(node_idx)
    current_nodes = set(target_set)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    for _ in range(num_hops):
        next_nodes = set(current_nodes)
        for s, d in zip(src, dst):
            if s in current_nodes or d in current_nodes:
                next_nodes.add(s)
                next_nodes.add(d)
        current_nodes = next_nodes

    subset_nodes = sorted(list(current_nodes))
    node_map = {n: i for i, n in enumerate(subset_nodes)}

    sub_src = []
    sub_dst = []
    edge_mask = []
    for s, d in zip(src, dst):
        if s in current_nodes and d in current_nodes:
            sub_src.append(node_map[s])
            sub_dst.append(node_map[d])
            edge_mask.append(True)
        else:
            edge_mask.append(False)

    sub_edge_index = (
        torch.tensor([sub_src, sub_dst], dtype=torch.long)
        if sub_src
        else torch.empty((2, 0), dtype=torch.long)
    )
    mapping = torch.tensor([node_map[n] for n in target_set if n in node_map], dtype=torch.long)

    return (
        torch.tensor(subset_nodes, dtype=torch.long),
        sub_edge_index,
        mapping,
        torch.tensor(edge_mask, dtype=torch.bool),
    )


def generate_node_embeddings(
    model: nn.Module,
    data: Data,
    device: str = "cpu",
) -> np.ndarray:
    """Generate node embeddings from a trained GNN model.

    Parameters
    ----------
    model : nn.Module
        Trained GraphSAGE, GAT, or GNN model with get_embeddings or forward method.
    data : Data
        PyG Data object containing x and edge_index.
    device : str
        Device to execute forward pass ('cpu' or 'cuda').

    Returns
    -------
    np.ndarray
        Array of node embeddings of shape [num_nodes, embedding_dim].
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(dev)
    model.eval()

    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)

    with torch.no_grad():
        if hasattr(model, "get_embeddings"):
            emb = model.get_embeddings(x, edge_index)
        else:
            emb = model(x, edge_index)

    return emb.detach().cpu().numpy()


def compute_graph_metrics(data: Data | Any) -> dict[str, Any]:
    """Compute topology and connectivity metrics for a graph.

    Parameters
    ----------
    data : Data | TransactionGraph
        PyG Data or TransactionGraph instance.

    Returns
    -------
    dict[str, Any]
        Dictionary with graph summary metrics (nodes, edges, density, degrees, etc.).
    """
    if hasattr(data, "edge_index") and hasattr(data, "num_nodes"):
        num_nodes = int(data.num_nodes)
        num_edges = int(data.edge_index.size(1))
        edge_index = data.edge_index
    elif hasattr(data, "nodes") and hasattr(data, "edges"):
        num_nodes = len(data.nodes)
        num_edges = sum(len(dests) for dests in data.edges.values())
        return data.summary()
    else:
        raise ValueError("Unsupported graph object passed to compute_graph_metrics")

    if num_nodes == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "max_degree": 0,
        }

    max_possible_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
    density = float(num_edges / max_possible_edges) if max_possible_edges > 0 else 0.0

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    out_degrees = np.bincount(src, minlength=num_nodes)
    in_degrees = np.bincount(dst, minlength=num_nodes)
    total_degrees = out_degrees + in_degrees

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": float(density),
        "avg_degree": float(np.mean(total_degrees)),
        "max_degree": int(np.max(total_degrees)),
        "min_degree": int(np.min(total_degrees)),
        "avg_in_degree": float(np.mean(in_degrees)),
        "avg_out_degree": float(np.mean(out_degrees)),
        "isolated_nodes_count": int(np.sum(total_degrees == 0)),
    }
