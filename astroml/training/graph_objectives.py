"""Pluggable loss and metric objectives for graph training (issue #738).

Every graph trainer in this package hard-codes both its loss and its notion of
"how well is this doing": ``train_gcn`` calls :func:`torch.nn.functional.nll_loss`
and computes accuracy inline, ``train_link_prediction`` delegates to the task
object, and ``train_sage`` reconstructs neighbour features. Adding a task means
copying a training loop, and comparing two tasks means comparing two ad-hoc
metric dictionaries.

An objective packages the pair — the loss a task optimises and the metrics that
describe it — behind one interface, so a training loop can be written once and
told which task it is running.

Two are provided:

* :class:`NodeClassificationObjective` — cross-entropy over class logits, with
  accuracy and macro-F1.
* :class:`LinkPredictionObjective` — binary cross-entropy over edge scores,
  with ROC-AUC and average precision.

Register your own with :func:`register_objective`.

Large graphs
------------
Metrics are computed in Torch on the tensors' own device. Nothing is moved to
NumPy and no pairwise matrix is materialised: ROC-AUC uses the rank identity
(O(n log n), one sort) rather than counting pairs (O(n²)), and average
precision is a single cumulative pass over the sorted scores. Above
``max_metric_samples`` the metric set is computed on a deterministic
subsample — a bounded, reproducible estimate rather than an unbounded sort of
every edge in the graph. The loss is never subsampled.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "GraphObjective",
    "LinkPredictionObjective",
    "NodeClassificationObjective",
    "available_objectives",
    "get_objective",
    "register_objective",
    "validate_edge_index",
    "validate_masks",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> None:
    """Check that ``edge_index`` is a usable COO edge list.

    An out-of-range index is the failure worth catching early: indexing a node
    feature matrix with it either raises deep inside a scatter kernel, where
    the message names neither the tensor nor the offending row, or — on CUDA —
    corrupts memory and produces a plausible-looking loss curve computed from
    garbage.

    Args:
        edge_index: Tensor shaped ``[2, E]``.
        num_nodes: Number of nodes the indices address.

    Raises:
        ValueError: If the shape, dtype or index range is wrong.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"edge_index must be an integer tensor, got {edge_index.dtype}")
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}")

    if edge_index.numel() == 0:
        return

    # `.min()`/`.max()` are two reductions over the edge list rather than a
    # Python loop, so this stays negligible next to a training step even on
    # graphs with tens of millions of edges.
    lowest = int(edge_index.min())
    highest = int(edge_index.max())
    if lowest < 0:
        raise ValueError(f"edge_index contains a negative node id ({lowest})")
    if highest >= num_nodes:
        raise ValueError(
            f"edge_index references node {highest} but the graph has {num_nodes} nodes"
        )


def validate_masks(masks: dict[str, torch.Tensor], num_nodes: int) -> None:
    """Check that named boolean node masks are well-formed and disjoint.

    Overlapping train and validation masks are the quiet version of a leak:
    training still converges, validation metrics still improve, and the
    reported numbers describe nodes the model was fitted on.

    Args:
        masks: Mapping of split name to a boolean tensor of length ``num_nodes``.
        num_nodes: Number of nodes in the graph.

    Raises:
        ValueError: If a mask has the wrong shape or dtype, or two masks overlap.
    """
    for name, mask in masks.items():
        if mask.dtype != torch.bool:
            raise ValueError(f"mask {name!r} must be a boolean tensor, got {mask.dtype}")
        if mask.dim() != 1 or mask.size(0) != num_nodes:
            raise ValueError(
                f"mask {name!r} must have shape [{num_nodes}], got {tuple(mask.shape)}"
            )

    names = sorted(masks)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = int((masks[left] & masks[right]).sum())
            if overlap:
                raise ValueError(
                    f"masks {left!r} and {right!r} overlap on {overlap} nodes; "
                    "validation metrics would be measured on training nodes"
                )


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _subsample(
    scores: torch.Tensor, targets: torch.Tensor, limit: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically take at most ``limit`` elements.

    Seeded from a generator on the tensor's own device rather than the global
    RNG: metric computation must not perturb the training stream, or the same
    run produces different weights depending on whether metrics were logged.
    """
    total = scores.numel()
    if limit <= 0 or total <= limit:
        return scores, targets

    generator = torch.Generator(device=scores.device)
    generator.manual_seed(seed)
    picked = torch.randperm(total, generator=generator, device=scores.device)[:limit]
    return scores[picked], targets[picked]


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC-AUC via the rank identity, in Torch.

    ``AUC = (sum of positive ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)``

    One sort, no pairwise comparison matrix, no host transfer. Ties are given
    their average rank, matching ``sklearn.metrics.roc_auc_score``.

    Returns ``0.5`` — the value of a coin flip — when one class is absent and
    the quantity is undefined.
    """
    positives = labels > 0.5
    n_pos = int(positives.sum())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float64)

    # Average the ranks within each group of equal scores, so that a model
    # emitting one constant score scores 0.5 rather than 0 or 1 depending on
    # sort order.
    sorted_scores = scores[order]
    unique, inverse, counts = torch.unique(sorted_scores, return_inverse=True, return_counts=True)
    if unique.numel() != sorted_scores.numel():
        rank_sums = torch.zeros(unique.numel(), device=scores.device, dtype=torch.float64)
        rank_sums.index_add_(0, inverse, ranks[order])
        averaged = (rank_sums / counts.to(torch.float64))[inverse]
        ranks[order] = averaged

    positive_rank_sum = float(ranks[positives].sum())
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Average precision, as a single cumulative pass over sorted scores.

    Returns ``0.0`` when there are no positives, where the quantity is
    undefined.
    """
    n_pos = int((labels > 0.5).sum())
    if n_pos == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    ordered_labels = (labels[order] > 0.5).to(torch.float64)

    true_positives = torch.cumsum(ordered_labels, dim=0)
    positions = torch.arange(
        1, ordered_labels.numel() + 1, device=scores.device, dtype=torch.float64
    )
    precision_at_k = true_positives / positions

    # Only the ranks holding a positive contribute a term.
    return float((precision_at_k * ordered_labels).sum() / n_pos)


def _macro_f1(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Unweighted mean per-class F1.

    Computed from a confusion count built with ``bincount`` over
    ``target * C + prediction``: one pass, no per-class Python loop, and no
    dependency on scikit-learn in the training loop.

    Classes absent from both prediction and target contribute 0, matching
    ``sklearn``'s default for macro averaging.
    """
    if num_classes <= 0:
        return 0.0

    pairs = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    confusion = torch.bincount(pairs, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )

    true_positive = confusion.diagonal().to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    actual = confusion.sum(dim=1).to(torch.float64)

    denominator = predicted + actual
    # 2·TP / (predicted + actual) is F1; a class nobody predicted and nobody
    # holds has denominator 0 and scores 0 rather than NaN.
    per_class = torch.where(
        denominator > 0,
        2 * true_positive / denominator.clamp(min=1e-12),
        torch.zeros_like(denominator),
    )
    return float(per_class.mean())


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class GraphObjective(ABC):
    """A task's loss together with the metrics that describe it.

    Subclasses define what a model's outputs mean. A training loop only needs
    :meth:`loss` to optimise and :meth:`metrics` to report, so the same loop
    serves node classification, link prediction and anything registered later.

    Attributes:
        name: Registry key.
        monitor: Metric a report should select the best epoch by.
        higher_is_better: Direction of ``monitor``.
        max_metric_samples: Cap on elements used for metric computation; ``0``
            disables subsampling. The loss always sees everything.
        seed: Seed for that subsample, so a rerun reports the same numbers.
    """

    name: str = "objective"
    monitor: str = "loss"
    higher_is_better: bool = False

    def __init__(self, *, max_metric_samples: int = 1_000_000, seed: int = 42) -> None:
        if max_metric_samples < 0:
            raise ValueError("max_metric_samples must be non-negative")
        self.max_metric_samples = max_metric_samples
        self.seed = seed

    @abstractmethod
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Differentiable loss for a batch. Returns a scalar tensor."""

    @abstractmethod
    def metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Plain-float metrics for a batch. Never differentiable."""

    def evaluate(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Loss and metrics together, detached, for a reporting step."""
        with torch.no_grad():
            report = {"loss": float(self.loss(outputs, targets))}
            report.update(self.metrics(outputs, targets))
        return report

    def is_better(self, candidate: float, incumbent: float) -> bool:
        """Whether ``candidate`` improves on ``incumbent`` under :attr:`monitor`."""
        return candidate > incumbent if self.higher_is_better else candidate < incumbent

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(monitor={self.monitor!r}, "
            f"higher_is_better={self.higher_is_better})"
        )


class NodeClassificationObjective(GraphObjective):
    """Multi-class classification over nodes.

    Args:
        log_input: Set when the model already applies ``log_softmax`` — which
            the GCN in this package does, so that its outputs feed
            :func:`~torch.nn.functional.nll_loss`. Passing raw logits to
            ``nll_loss`` trains without complaint and converges to nonsense,
            so this is explicit rather than inferred.
        class_weight: Optional per-class loss weighting for imbalanced labels.
    """

    name = "node_classification"
    monitor = "accuracy"
    higher_is_better = True

    def __init__(
        self,
        *,
        log_input: bool = True,
        class_weight: torch.Tensor | None = None,
        max_metric_samples: int = 1_000_000,
        seed: int = 42,
    ) -> None:
        super().__init__(max_metric_samples=max_metric_samples, seed=seed)
        self.log_input = log_input
        self.class_weight = class_weight

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._check(outputs, targets)
        weight = self.class_weight
        if weight is not None:
            weight = weight.to(outputs.device)
        if self.log_input:
            return F.nll_loss(outputs, targets, weight=weight)
        return F.cross_entropy(outputs, targets, weight=weight)

    def metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        self._check(outputs, targets)
        if outputs.numel() == 0:
            return {"accuracy": 0.0, "macro_f1": 0.0}

        num_classes = outputs.size(1)
        with torch.no_grad():
            predictions = outputs.argmax(dim=1)
            predictions, targets = _subsample(
                predictions, targets, self.max_metric_samples, self.seed
            )
            accuracy = float((predictions == targets).to(torch.float64).mean())
            macro_f1 = _macro_f1(predictions, targets, num_classes)

        return {"accuracy": accuracy, "macro_f1": macro_f1}

    @staticmethod
    def _check(outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if outputs.dim() != 2:
            raise ValueError(
                f"node classification outputs must have shape [N, C], got {tuple(outputs.shape)}"
            )
        if targets.dim() != 1 or targets.size(0) != outputs.size(0):
            raise ValueError(
                f"targets must have shape [{outputs.size(0)}], got {tuple(targets.shape)}"
            )


class LinkPredictionObjective(GraphObjective):
    """Binary classification over candidate edges.

    ``outputs`` are raw edge scores (logits) and ``targets`` are 1 for an edge
    that exists and 0 for a sampled negative. Loss is binary cross-entropy
    computed from the logits directly — going through an explicit sigmoid first
    loses the log-sum-exp stabilisation and overflows on confident scores.

    Args:
        pos_weight: Weight on the positive class, for the usual case where
            negatives are sampled at a multiple of the positives.
    """

    name = "link_prediction"
    monitor = "auc"
    higher_is_better = True

    def __init__(
        self,
        *,
        pos_weight: float | None = None,
        max_metric_samples: int = 1_000_000,
        seed: int = 42,
    ) -> None:
        super().__init__(max_metric_samples=max_metric_samples, seed=seed)
        self.pos_weight = pos_weight

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._check(outputs, targets)
        weight = (
            None
            if self.pos_weight is None
            else torch.tensor(self.pos_weight, device=outputs.device, dtype=outputs.dtype)
        )
        return F.binary_cross_entropy_with_logits(
            outputs, targets.to(outputs.dtype), pos_weight=weight
        )

    def metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        self._check(outputs, targets)
        if outputs.numel() == 0:
            return {"auc": 0.5, "average_precision": 0.0, "accuracy": 0.0}

        with torch.no_grad():
            scores, labels = _subsample(
                outputs.detach(), targets, self.max_metric_samples, self.seed
            )
            labels = labels.to(torch.float64)
            # Ranking metrics are invariant under the sigmoid, so scoring the
            # logits directly saves a pass over every edge.
            auc = _roc_auc(scores, labels)
            average_precision = _average_precision(scores, labels)
            # A logit above zero is a probability above one half.
            accuracy = float(((scores > 0).to(torch.float64) == labels).to(torch.float64).mean())

        return {"auc": auc, "average_precision": average_precision, "accuracy": accuracy}

    @staticmethod
    def _check(outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if outputs.dim() != 1:
            raise ValueError(
                f"link prediction outputs must have shape [E], got {tuple(outputs.shape)}"
            )
        if targets.shape != outputs.shape:
            raise ValueError(
                f"targets must have shape {tuple(outputs.shape)}, got {tuple(targets.shape)}"
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., GraphObjective]] = {
    NodeClassificationObjective.name: NodeClassificationObjective,
    LinkPredictionObjective.name: LinkPredictionObjective,
}


def register_objective(name: str, factory: Callable[..., GraphObjective]) -> None:
    """Register an objective under ``name``.

    Re-registering an existing name replaces it and logs, so a plugin
    overriding a built-in leaves a trace rather than doing it silently.
    """
    if not name:
        raise ValueError("objective name must be a non-empty string")
    if name in _REGISTRY:
        logger.warning("Replacing already-registered graph objective %r", name)
    _REGISTRY[name] = factory


def get_objective(name: str, **kwargs: object) -> GraphObjective:
    """Construct a registered objective by name.

    Raises:
        KeyError: If ``name`` is unknown; the message lists what is available.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown graph objective {name!r}; available: {', '.join(available_objectives())}"
        ) from None
    return factory(**kwargs)  # type: ignore[arg-type]


def available_objectives() -> Iterable[str]:
    """Names of every registered objective, sorted."""
    return sorted(_REGISTRY)
