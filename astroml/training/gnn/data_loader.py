"""Graph data loaders and preprocessing pipelines for transaction graph analysis.

Provides datasets, neighbor samplers, batch loaders, and graph feature
standardization pipelines for GNN training and evaluation.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = Any  # type: ignore[misc,assignment]
    NeighborLoader = None

from astroml.features.gnn.sampler import MultiHopSampler


class GraphDataPreprocessor:
    """Preprocessing and feature scaling pipeline for graph datasets."""

    def __init__(
        self,
        normalize_features: bool = True,
        scale_edge_weights: bool = True,
    ) -> None:
        self.normalize_features = normalize_features
        self.scale_edge_weights = scale_edge_weights
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray | torch.Tensor) -> GraphDataPreprocessor:
        """Compute mean and standard deviation for feature normalization."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)
        self.std_[self.std_ == 0.0] = 1.0  # Prevent divide by zero
        return self

    def transform(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Apply standardization to node features."""
        is_torch = isinstance(x, torch.Tensor)
        np_x = x.detach().cpu().numpy() if is_torch else np.asarray(x, dtype=np.float32)

        if self.normalize_features:
            if self.mean_ is None or self.std_ is None:
                self.fit(np_x)
            assert self.mean_ is not None and self.std_ is not None
            np_x = (np_x - self.mean_) / self.std_

        return torch.tensor(np_x, dtype=torch.float32)

    def fit_transform(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Fit scaler and transform features."""
        return self.fit(x).transform(x)


def create_node_masks(
    num_nodes: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    labels: np.ndarray | torch.Tensor | None = None,
    stratify: bool = False,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate boolean train, validation, and test masks for graph nodes.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes.
    train_ratio : float
        Proportion for training (default 0.7).
    val_ratio : float
        Proportion for validation (default 0.15).
    test_ratio : float
        Proportion for testing (default 0.15).
    labels : np.ndarray | torch.Tensor | None
        Optional node labels for stratified splitting.
    stratify : bool
        Whether to perform stratified sampling based on labels.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (train_mask, val_mask, test_mask) as boolean tensors of shape [num_nodes].
    """
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    rng = np.random.default_rng(seed)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    if stratify and labels is not None:
        np_labels = (
            labels.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor)
            else np.asarray(labels)
        )
        unique_classes = np.unique(np_labels)
        for cls in unique_classes:
            cls_indices = np.where(np_labels == cls)[0]
            rng.shuffle(cls_indices)
            n_cls = len(cls_indices)
            n_train = max(1, int(n_cls * train_ratio))
            n_val = max(1, int(n_cls * val_ratio)) if val_ratio > 0 and n_cls > 2 else 0

            train_idx = cls_indices[:n_train]
            val_idx = cls_indices[n_train : n_train + n_val]
            test_idx = cls_indices[n_train + n_val :]

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
    else:
        indices = np.arange(num_nodes)
        rng.shuffle(indices)
        n_train = int(num_nodes * train_ratio)
        n_val = int(num_nodes * val_ratio)

        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train : n_train + n_val]] = True
        test_mask[indices[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask


class TransactionGraphDataset(Dataset):
    """PyTorch Dataset wrapping a transaction graph representation."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        node_ids: Sequence[str] | None = None,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.edge_attr = edge_attr
        self.node_ids = (
            list(node_ids) if node_ids is not None else [str(i) for i in range(x.size(0))]
        )

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "node_idx": idx,
            "node_id": self.node_ids[idx],
            "feature": self.x[idx],
        }
        if self.y is not None:
            item["label"] = self.y[idx]
        return item

    def get_pyg_data(self) -> Any:
        """Export as PyG Data object."""
        if _HAS_PYG:
            return Data(
                x=self.x,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=self.y,
                num_nodes=self.x.size(0),
            )
        return {
            "x": self.x,
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "y": self.y,
            "num_nodes": self.x.size(0),
        }


class TransactionGraphDataLoader:
    """Mini-batch loader for neighbor sampling on large transaction networks."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor | None = None,
        target_nodes: torch.Tensor | Sequence[int] | None = None,
        batch_size: int = 256,
        num_neighbors: list[int] | None = None,
        shuffle: bool = True,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [15, 10]
        self.shuffle = shuffle

        if target_nodes is None:
            self.target_nodes = torch.arange(x.size(0), dtype=torch.long)
        elif isinstance(target_nodes, torch.Tensor):
            self.target_nodes = target_nodes.long()
        else:
            self.target_nodes = torch.tensor(list(target_nodes), dtype=torch.long)

        self.sampler = MultiHopSampler(
            edge_index=edge_index,
            num_nodes=x.size(0),
            fanout=self.num_neighbors,
        )

    def __len__(self) -> int:
        return (len(self.target_nodes) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        nodes = self.target_nodes
        if self.shuffle:
            perm = torch.randperm(len(nodes))
            nodes = nodes[perm]

        for i in range(0, len(nodes), self.batch_size):
            batch_targets = nodes[i : i + self.batch_size]
            adjs, sampled_node_ids = self.sampler.sample(batch_targets)

            batch_x = self.x[sampled_node_ids]
            batch_y = self.y[batch_targets] if self.y is not None else None

            yield {
                "target_nodes": batch_targets,
                "sampled_nodes": sampled_node_ids,
                "x": batch_x,
                "y": batch_y,
                "adjs": adjs,
            }
