"""Self-supervised link prediction training task.

Training objective
------------------
Given a stream of timestamped transactions (edges) between accounts (nodes):

1. Take a **context window** ending at ledger ``L``.
2. The **positive set** = edges that appear within the next ``N`` ledgers
   (the "future" window).
3. The **negative set** = the same number of (u, v) pairs sampled uniformly
   from account pairs that did *not* transact in the future window.
4. Train :class:`~astroml.models.link_prediction.LinkPredictor` to
   distinguish positives from negatives using binary cross-entropy.

This is a **self-supervised** objective: no manual labels are needed —
the future transaction graph itself provides supervision.

Key classes
-----------
* :class:`LedgerSplit` — dataclass holding one context/future pair of edge
  sets, plus the node index mapping needed to build ``edge_index`` tensors.
* :class:`LinkPredictionTask` — orchestrates splitting, negative sampling,
  and per-step training.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from astroml.features.graph.snapshot import Edge, window_snapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LedgerSplit:
    """One context / future pair used for a single training step.

    Attributes:
        context_edges: Edges used as the input graph (message-passing graph).
        future_edges: Edges observed in the next N ledgers (positive labels).
        node_index: Mapping from account-id string → contiguous integer index.
    """
    context_edges: List[Edge]
    future_edges: List[Edge]
    node_index: Dict[str, int]

    @property
    def num_nodes(self) -> int:
        return len(self.node_index)

    def to_edge_index(self, edges: List[Edge], device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Convert an edge list to a PyG-style ``[2, E]`` LongTensor.

        Unknown accounts (not in ``node_index``) are silently skipped.
        """
        pairs = [
            (self.node_index[e.src], self.node_index[e.dst])
            for e in edges
            if e.src in self.node_index and e.dst in self.node_index
        ]
        if not pairs:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        src, dst = zip(*pairs)
        return torch.tensor([list(src), list(dst)], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Negative edge sampling
# ---------------------------------------------------------------------------

def sample_negative_edges(
    num_nodes: int,
    positive_set: Set[Tuple[int, int]],
    num_samples: int,
    max_attempts: int = 10,
    rng: Optional[random.Random] = None,
) -> List[Tuple[int, int]]:
    """Sample (u, v) pairs that are not in *positive_set*.

    Samples without replacement up to *num_samples* unique non-edges.
    Gives up after ``num_samples * max_attempts`` draws to avoid infinite
    loops on very dense graphs, returning however many non-edges were found.

    Args:
        num_nodes: Total number of nodes.
        positive_set: Set of ``(src, dst)`` integer pairs to exclude.
        num_samples: Desired number of negative samples.
        max_attempts: Multiplier for the draw budget.
        rng: Optional seeded :class:`random.Random` instance for
            reproducibility.

    Returns:
        List of ``(src, dst)`` integer pairs.
    """
    if rng is None:
        rng = random.Random()

    negatives: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set(positive_set)
    budget = num_samples * max_attempts

    while len(negatives) < num_samples and budget > 0:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        budget -= 1
        if u == v or (u, v) in seen:
            continue
        negatives.append((u, v))
        seen.add((u, v))

    return negatives


# ---------------------------------------------------------------------------
# Main task
# ---------------------------------------------------------------------------

class LinkPredictionTask:
    """Self-supervised link prediction over a Stellar ledger stream.

    Splits a sorted edge sequence into overlapping (context, future) windows
    keyed by ledger sequence number.  For each window a training step is run
    that optimises :class:`~astroml.models.link_prediction.LinkPredictor`
    to predict whether two accounts will transact in the next ``n_future``
    ledgers.

    Args:
        edges: Full edge sequence sorted by ``timestamp`` (ledger sequence).
        n_future: Number of ledgers ahead to use as the positive label window.
        context_ledgers: Number of ledgers in each context window.  If
            ``None`` all edges before the future window are used as context.
        neg_sampling_ratio: Ratio of negative to positive edges per step.
        device: PyTorch device for tensor operations.
        seed: Random seed for reproducible negative sampling.

    Example::

        task = LinkPredictionTask(edges, n_future=10, context_ledgers=100)
        splits = task.build_splits()
        for split in splits:
            loss = task.train_step(model, optimizer, split, node_features)
    """

    def __init__(
        self,
        edges: Sequence[Edge],
        n_future: int = 10,
        context_ledgers: Optional[int] = None,
        neg_sampling_ratio: float = 1.0,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ) -> None:
        if n_future < 1:
            raise ValueError(f"n_future must be >= 1, got {n_future}")
        if neg_sampling_ratio <= 0:
            raise ValueError(f"neg_sampling_ratio must be > 0, got {neg_sampling_ratio}")

        self.edges = sorted(edges, key=lambda e: e.timestamp)
        self.n_future = n_future
        self.context_ledgers = context_ledgers
        self.neg_sampling_ratio = neg_sampling_ratio
        self.device = device
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Build splits
    # ------------------------------------------------------------------

    def build_splits(self) -> List[LedgerSplit]:
        """Enumerate non-overlapping (context, future) window pairs.

        Iterates over unique ledger timestamps, using each as the boundary
        between context and future:

        * **context** = edges with ``timestamp < boundary``  (optionally
          capped to the last ``context_ledgers`` distinct ledgers)
        * **future** = edges with ``boundary <= timestamp < boundary + n_future``

        Pairs where either partition is empty are skipped.

        Returns:
            List of :class:`LedgerSplit` objects, one per boundary ledger.
        """
        if not self.edges:
            return []

        ledger_seqs = sorted({e.timestamp for e in self.edges})
        splits: List[LedgerSplit] = []

        for i, boundary in enumerate(ledger_seqs):
            future_end = boundary + self.n_future

            context_edges = [e for e in self.edges if e.timestamp < boundary]
            future_edges = [e for e in self.edges if boundary <= e.timestamp < future_end]

            if not context_edges or not future_edges:
                continue

            # Optionally restrict context to last N ledgers.
            if self.context_ledgers is not None:
                context_seqs = sorted({e.timestamp for e in context_edges})
                if len(context_seqs) > self.context_ledgers:
                    cutoff_seq = context_seqs[-self.context_ledgers]
                    context_edges = [e for e in context_edges if e.timestamp >= cutoff_seq]

            # Build a shared node index across both windows.
            accounts: Set[str] = set()
            for e in context_edges + future_edges:
                accounts.add(e.src)
                accounts.add(e.dst)
            node_index = {acc: idx for idx, acc in enumerate(sorted(accounts))}

            splits.append(LedgerSplit(
                context_edges=context_edges,
                future_edges=future_edges,
                node_index=node_index,
            ))

        logger.info("Built %d link-prediction splits (n_future=%d)", len(splits), self.n_future)
        return splits

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def sample_negatives(self, split: LedgerSplit) -> torch.Tensor:
        """Sample negative edges for *split*.

        Returns a ``[2, E_neg]`` LongTensor of (src, dst) pairs that do not
        appear in ``split.future_edges``.
        """
        n_pos = max(1, len(split.future_edges))
        n_neg = max(1, int(n_pos * self.neg_sampling_ratio))

        pos_set: Set[Tuple[int, int]] = set()
        for e in split.future_edges:
            if e.src in split.node_index and e.dst in split.node_index:
                pos_set.add((split.node_index[e.src], split.node_index[e.dst]))

        neg_pairs = sample_negative_edges(
            num_nodes=split.num_nodes,
            positive_set=pos_set,
            num_samples=n_neg,
            rng=self._rng,
        )

        if not neg_pairs:
            return torch.zeros(2, 0, dtype=torch.long, device=self.device)

        src, dst = zip(*neg_pairs)
        return torch.tensor([list(src), list(dst)], dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        split: LedgerSplit,
        node_features: torch.Tensor,
    ) -> float:
        """Run one gradient update step on *split*.

        Args:
            model: :class:`~astroml.models.link_prediction.LinkPredictor`
                instance.
            optimizer: Torch optimizer (e.g. Adam).
            split: One :class:`LedgerSplit` produced by :meth:`build_splits`.
            node_features: Node feature matrix ``[split.num_nodes, F]`` on
                the correct device.

        Returns:
            Scalar loss value for this step.
        """
        model.train()
        optimizer.zero_grad()

        context_edge_index = split.to_edge_index(split.context_edges, device=self.device)
        pos_edge_index = split.to_edge_index(split.future_edges, device=self.device)
        neg_edge_index = self.sample_negatives(split)

        if pos_edge_index.size(1) == 0:
            return 0.0

        z = model.encode(node_features, context_edge_index)
        loss = model.loss(z, pos_edge_index, neg_edge_index)
        loss.backward()
        optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: nn.Module,
        split: LedgerSplit,
        node_features: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate link prediction on *split*.

        Computes:
        * **auc** — area under the ROC curve
        * **avg_precision** — average precision (area under PR curve)

        Args:
            model: Trained :class:`~astroml.models.link_prediction.LinkPredictor`.
            split: A held-out :class:`LedgerSplit`.
            node_features: Node feature matrix ``[split.num_nodes, F]``.

        Returns:
            Dict with ``"auc"`` and ``"avg_precision"`` keys.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        model.eval()
        with torch.no_grad():
            context_edge_index = split.to_edge_index(split.context_edges, device=self.device)
            pos_edge_index = split.to_edge_index(split.future_edges, device=self.device)
            neg_edge_index = self.sample_negatives(split)

            z = model.encode(node_features, context_edge_index)

            pos_scores = torch.sigmoid(model.decode(z, pos_edge_index)).cpu().numpy()
            neg_scores = torch.sigmoid(model.decode(z, neg_edge_index)).cpu().numpy()

        import numpy as np
        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([
            np.ones(len(pos_scores)),
            np.zeros(len(neg_scores)),
        ])

        metrics: Dict[str, float] = {}
        try:
            metrics["auc"] = float(roc_auc_score(labels, scores))
        except ValueError:
            metrics["auc"] = 0.5

        try:
            metrics["avg_precision"] = float(average_precision_score(labels, scores))
        except ValueError:
            metrics["avg_precision"] = 0.0

        return metrics
