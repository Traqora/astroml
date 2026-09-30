"""Connected subgraph sampling for mini-batch training — issue #734.

A snapshot of the Stellar network does not fit in a GPU, and slicing it by
taking a random set of nodes gives a batch that is mostly isolated vertices:
the edges that made it a graph are exactly the ones the slice cut. This
module samples *connected* subgraphs of bounded size instead, so every batch
is a piece of the network a message-passing layer can actually propagate
through.

Two strategies, because they are biased differently and the right choice
depends on the task:

``bfs`` grows outward ring by ring, so the sample is the seed's dense local
neighbourhood — good for node classification, and heavily biased towards
high-degree regions.

``random_walk`` follows edges with restarts, reaching further from the seed
for the same node budget and sampling paths rather than balls — less biased
towards hubs, at the cost of a patchier local picture.

``degree_bias`` tunes the remaining freedom: 0.0 picks among a node's
neighbours uniformly, 1.0 picks in proportion to degree, so a caller can
deliberately over- or under-sample the hubs.

Everything is seeded. The same sampler, budget and seed give the same
subgraph on any machine, which is what makes a training run reproducible
rather than merely repeatable. Adjacency is built once in the constructor and
shared by every batch, so sampling a thousand mini-batches costs one pass
over the edges plus the walks themselves.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: How a subgraph is grown from its seed.
STRATEGIES = ("bfs", "random_walk")

#: How the seed node itself is chosen when the caller does not name one.
SEED_STRATEGIES = ("uniform", "degree")

_DEFAULT_RESTART_PROBABILITY = 0.15


class InvalidSubgraphRequestError(ValueError):
    """Raised when a subgraph cannot be sampled as asked."""


@dataclass(frozen=True)
class Subgraph:
    """A connected slice of a snapshot."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    seed_node: str
    strategy: str
    #: True when the node budget stopped the walk before the component was
    #: exhausted, so the sample is a bounded view rather than a whole island.
    truncated: bool

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def to_edge_index(self) -> tuple[list[int], list[int]]:
        """``(row, col)`` with local indices into :attr:`nodes`."""
        index_of = {node: i for i, node in enumerate(self.nodes)}
        return (
            [index_of[src] for src, _ in self.edges],
            [index_of[dst] for _, dst in self.edges],
        )


def _edge_endpoints(edge: Any) -> tuple[str, str]:
    if isinstance(edge, Mapping):
        src = edge.get("src", edge.get("source"))
        dst = edge.get("dst", edge.get("destination"))
    else:
        src = getattr(edge, "src", None)
        dst = getattr(edge, "dst", None)
    if src is None or dst is None:
        raise InvalidSubgraphRequestError(f"edge is missing src or dst: {edge!r}")
    return str(src), str(dst)


class SubgraphSampler:
    """Samples connected subgraphs from a fixed snapshot.

    The adjacency is built once here and reused, so the per-batch cost is
    proportional to the batch, not to the snapshot. Instantiate one sampler
    per snapshot and pull as many mini-batches from it as training needs.
    """

    def __init__(self, edges: Iterable[Any], directed: bool = False) -> None:
        """Args:
        edges: Edge records — ``Edge`` instances or mappings.
        directed: When true the walk only follows edges forwards. The
            default treats edges as traversable both ways, because a
            batch that can only move downstream tends to dead-end
            immediately on a payment graph.
        """
        self.directed = directed

        pairs: set[tuple[str, str]] = set()
        neighbours: dict[str, set[str]] = {}

        for edge in edges:
            src, dst = _edge_endpoints(edge)
            pairs.add((src, dst))
            neighbours.setdefault(src, set())
            neighbours.setdefault(dst, set())
            if src != dst:
                neighbours[src].add(dst)
                if not directed:
                    neighbours[dst].add(src)

        # Sorted tuples: the walk needs a fixed iteration order for a seed to
        # reproduce, and they are cheaper to traverse repeatedly than sets.
        self._adjacency: dict[str, tuple[str, ...]] = {
            node: tuple(sorted(peers)) for node, peers in neighbours.items()
        }
        self._pairs = pairs
        self._nodes: tuple[str, ...] = tuple(sorted(neighbours))
        self._degree = {node: len(peers) for node, peers in self._adjacency.items()}

    @property
    def nodes(self) -> tuple[str, ...]:
        return self._nodes

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def degree(self, node: str) -> int:
        return self._degree.get(node, 0)

    # -- seeding ---------------------------------------------------------

    def _choose_seed(self, rng: random.Random, seed_strategy: str) -> str:
        if not self._nodes:
            raise InvalidSubgraphRequestError("cannot sample from an empty graph")
        if seed_strategy == "uniform":
            return self._nodes[rng.randrange(len(self._nodes))]
        # "degree": proportional to degree, which concentrates batches on the
        # busy parts of the network — appropriate when the label of interest
        # lives there, misleading otherwise.
        weights = [self._degree[node] + 1 for node in self._nodes]
        return rng.choices(self._nodes, weights=weights, k=1)[0]

    def _pick_neighbour(
        self,
        candidates: Sequence[str],
        rng: random.Random,
        degree_bias: float,
    ) -> str:
        if degree_bias <= 0.0:
            return candidates[rng.randrange(len(candidates))]
        # A degree_bias of 1 gives selection proportional to degree; values in
        # between interpolate through the exponent, so the knob is smooth
        # rather than a switch between two behaviours.
        weights = [(self._degree.get(node, 0) + 1) ** degree_bias for node in candidates]
        return rng.choices(candidates, weights=weights, k=1)[0]

    # -- sampling --------------------------------------------------------

    def sample(
        self,
        max_nodes: int,
        seed_node: str | None = None,
        strategy: str = "bfs",
        seed: int | None = None,
        seed_strategy: str = "uniform",
        degree_bias: float = 0.0,
        restart_probability: float = _DEFAULT_RESTART_PROBABILITY,
        max_steps: int | None = None,
    ) -> Subgraph:
        """Sample one connected subgraph.

        Args:
            max_nodes: Hard cap on the number of nodes. The walk stops here
                even if the component is larger — that bound is the point.
            seed_node: Where to start. Chosen by ``seed_strategy`` when
                omitted.
            strategy: ``"bfs"`` or ``"random_walk"``.
            seed: RNG seed. Two calls with the same seed and arguments
                produce the same subgraph.
            seed_strategy: ``"uniform"`` or ``"degree"``.
            degree_bias: 0.0 picks neighbours uniformly, 1.0 in proportion to
                degree.
            restart_probability: Random-walk only — chance per step of
                jumping back to the seed, which keeps the walk local instead
                of drifting away.
            max_steps: Random-walk only — step budget, so a walk through a
                small component cannot spin. Defaults to a multiple of
                ``max_nodes``.

        Returns:
            A :class:`Subgraph` containing every edge of the snapshot whose
            endpoints both fell inside the sample.
        """
        if max_nodes < 1:
            raise InvalidSubgraphRequestError(f"max_nodes must be >= 1, got {max_nodes}")
        if strategy not in STRATEGIES:
            raise InvalidSubgraphRequestError(
                f"unknown strategy {strategy!r}; expected one of {STRATEGIES}"
            )
        if seed_strategy not in SEED_STRATEGIES:
            raise InvalidSubgraphRequestError(
                f"unknown seed_strategy {seed_strategy!r}; expected one of {SEED_STRATEGIES}"
            )
        if degree_bias < 0.0:
            raise InvalidSubgraphRequestError(f"degree_bias must be >= 0, got {degree_bias}")
        if not 0.0 <= restart_probability < 1.0:
            raise InvalidSubgraphRequestError(
                f"restart_probability must be in [0, 1), got {restart_probability}"
            )

        rng = random.Random(seed)

        if seed_node is None:
            seed_node = self._choose_seed(rng, seed_strategy)
        elif seed_node not in self._adjacency:
            raise InvalidSubgraphRequestError(f"seed node {seed_node!r} is not in this graph")

        if strategy == "bfs":
            selected, truncated = self._grow_bfs(seed_node, max_nodes, rng, degree_bias)
        else:
            selected, truncated = self._grow_random_walk(
                seed_node, max_nodes, rng, degree_bias, restart_probability, max_steps
            )

        nodes = tuple(sorted(selected))
        inside = set(nodes)
        edges = tuple(
            sorted((src, dst) for src, dst in self._pairs if src in inside and dst in inside)
        )

        return Subgraph(
            nodes=nodes,
            edges=edges,
            seed_node=seed_node,
            strategy=strategy,
            truncated=truncated,
        )

    def _grow_bfs(
        self,
        seed_node: str,
        max_nodes: int,
        rng: random.Random,
        degree_bias: float,
    ) -> tuple[set[str], bool]:
        selected = {seed_node}
        frontier = [seed_node]

        while frontier and len(selected) < max_nodes:
            next_frontier: list[str] = []
            for node in frontier:
                candidates = [n for n in self._adjacency.get(node, ()) if n not in selected]
                # Always randomise the ring before walking it. The adjacency
                # is stored sorted, so taking it as-is would make truncation
                # at max_nodes keep whichever neighbours sort first — an
                # alphabetical sample dressed up as an arbitrary one.
                candidates = self._ring_order(candidates, rng, degree_bias)
                for neighbour in candidates:
                    if len(selected) >= max_nodes:
                        return selected, True
                    selected.add(neighbour)
                    next_frontier.append(neighbour)
            frontier = next_frontier

        # Truncated only if something reachable was left out.
        reachable_remains = any(
            neighbour not in selected
            for node in selected
            for neighbour in self._adjacency.get(node, ())
        )
        return selected, reachable_remains

    def _ring_order(
        self,
        candidates: Sequence[str],
        rng: random.Random,
        degree_bias: float,
    ) -> list[str]:
        """Randomise a frontier ring, optionally weighted by degree."""
        ordered = list(candidates)
        if degree_bias <= 0.0:
            # Uniform, and O(k) — the weighted path below is quadratic, which
            # is not something to pay on a hub with thousands of neighbours
            # when no bias was asked for.
            rng.shuffle(ordered)
            return ordered

        remaining = ordered
        weighted: list[str] = []
        while remaining:
            chosen = self._pick_neighbour(remaining, rng, degree_bias)
            weighted.append(chosen)
            remaining.remove(chosen)
        return weighted

    def _grow_random_walk(
        self,
        seed_node: str,
        max_nodes: int,
        rng: random.Random,
        degree_bias: float,
        restart_probability: float,
        max_steps: int | None,
    ) -> tuple[set[str], bool]:
        selected = {seed_node}
        current = seed_node
        # Without a step cap, a walk over a component smaller than max_nodes
        # would keep stepping forever having already collected everything.
        budget = max_steps if max_steps is not None else max(64, max_nodes * 20)

        for _ in range(budget):
            if len(selected) >= max_nodes:
                return selected, True

            candidates = self._adjacency.get(current, ())
            if not candidates:
                current = seed_node
                continue

            current = self._pick_neighbour(candidates, rng, degree_bias)
            selected.add(current)

            if rng.random() < restart_probability:
                current = seed_node

        reachable_remains = any(
            neighbour not in selected
            for node in selected
            for neighbour in self._adjacency.get(node, ())
        )
        return selected, reachable_remains

    # -- batching --------------------------------------------------------

    def minibatches(
        self,
        max_nodes: int,
        num_batches: int,
        seed: int | None = None,
        strategy: str = "bfs",
        seed_strategy: str = "uniform",
        degree_bias: float = 0.0,
        distinct_seeds: bool = True,
        **kwargs: Any,
    ) -> list[Subgraph]:
        """Sample a series of subgraphs for one training epoch.

        Args:
            distinct_seeds: Draw each batch from a different seed node
                (sampling without replacement while nodes last). Without it
                the same dense region can be drawn repeatedly, which quietly
                turns an epoch into training on one neighbourhood.

        Every batch derives its RNG from ``seed``, so the whole epoch is
        reproducible from that one number.
        """
        if num_batches < 1:
            raise InvalidSubgraphRequestError(f"num_batches must be >= 1, got {num_batches}")
        if not self._nodes:
            raise InvalidSubgraphRequestError("cannot sample from an empty graph")

        rng = random.Random(seed)
        batches: list[Subgraph] = []
        used: set[str] = set()

        for index in range(num_batches):
            seed_node = None
            if distinct_seeds:
                available = [node for node in self._nodes if node not in used]
                if not available:
                    # Every node has seeded a batch; start the cycle again
                    # rather than returning fewer batches than asked for.
                    used.clear()
                    available = list(self._nodes)
                seed_node = self._choose_seed_from(available, rng, seed_strategy)
                used.add(seed_node)

            batches.append(
                self.sample(
                    max_nodes=max_nodes,
                    seed_node=seed_node,
                    strategy=strategy,
                    seed=rng.randrange(2**31) if seed is not None else None,
                    seed_strategy=seed_strategy,
                    degree_bias=degree_bias,
                    **kwargs,
                )
            )
            del index

        return batches

    def _choose_seed_from(
        self,
        available: Sequence[str],
        rng: random.Random,
        seed_strategy: str,
    ) -> str:
        if seed_strategy == "uniform":
            return available[rng.randrange(len(available))]
        weights = [self._degree.get(node, 0) + 1 for node in available]
        return rng.choices(available, weights=weights, k=1)[0]


def sample_connected_subgraph(
    edges: Iterable[Any],
    max_nodes: int,
    **kwargs: Any,
) -> Subgraph:
    """One-shot convenience wrapper around :class:`SubgraphSampler`.

    Builds the adjacency, takes a single sample and throws it away. For more
    than one batch construct a :class:`SubgraphSampler` and reuse it — that
    is the whole reason the adjacency lives on the object.
    """
    directed = bool(kwargs.pop("directed", False))
    return SubgraphSampler(edges, directed=directed).sample(max_nodes, **kwargs)


__all__ = [
    "SEED_STRATEGIES",
    "STRATEGIES",
    "InvalidSubgraphRequestError",
    "Subgraph",
    "SubgraphSampler",
    "sample_connected_subgraph",
]
