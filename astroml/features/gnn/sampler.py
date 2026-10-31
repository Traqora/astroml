from __future__ import annotations

import torch

from astroml.features.gnn.sage import sample_neighbors


class MultiHopSampler:
    """K-hop neighbor sampler with configurable fanout per layer.

    Parameters
    ----------
    edge_index : Tensor [2, E]
        Full graph edge index.
    num_nodes : int
        Total number of nodes in the graph.
    fanout : list[int]
        Number of neighbors to sample at each hop. Length determines number of hops.
        Example: [25, 10] means 25 neighbors at hop-1, 10 at hop-2.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        fanout: list[int],
    ) -> None:
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.fanout = fanout

    def sample(
        self, target_nodes: torch.Tensor
    ) -> tuple[list[tuple[torch.Tensor, tuple[int, int]]], torch.Tensor]:
        """Sample multi-hop neighborhood around target_nodes.

        Returns
        -------
        adjs : list of (edge_index, (num_src, num_dst))
            One entry per hop, ordered from outermost hop to innermost.
            edge_index uses local indices into ``all_nodes``. For every
            entry, source indices are ``< num_src`` and destination indices
            are ``< num_dst``, and the destination nodes are the first
            ``num_dst`` entries of ``all_nodes``.
        all_nodes : Tensor
            Union of all node IDs touched, ordered so that target_nodes
            appear at indices [0..len(target_nodes)-1], followed by hop-1
            neighbors, then hop-2 neighbors, and so on. Duplicate target IDs
            are collapsed to their first occurrence.
        """
        # Hop-by-hop node lists. Each list extends the previous one, so the
        # first ``len(previous)`` entries of a hop are exactly the previous
        # hop's nodes. That prefix property is what lets the encoder take
        # ``x[:num_dst]`` as the destination rows of every layer.
        layer_nodes: list[int] = []
        for n in target_nodes.tolist():
            if n not in layer_nodes:
                layer_nodes.append(int(n))
        local = {n: i for i, n in enumerate(layer_nodes)}

        inner_to_outer: list[tuple[torch.Tensor, tuple[int, int]]] = []
        for num_samples in self.fanout:
            num_dst = len(layer_nodes)
            dst_nodes = torch.tensor(layer_nodes, dtype=torch.long)
            src, dst = sample_neighbors(self.edge_index, dst_nodes, num_samples)

            for s in src.tolist():
                if s not in local:
                    local[s] = len(layer_nodes)
                    layer_nodes.append(int(s))

            if src.numel() == 0:
                local_edge = torch.zeros((2, 0), dtype=torch.long)
            else:
                local_src = torch.tensor([local[int(s)] for s in src.tolist()], dtype=torch.long)
                local_dst = torch.tensor([local[int(d)] for d in dst.tolist()], dtype=torch.long)
                local_edge = torch.stack([local_src, local_dst])

            inner_to_outer.append((local_edge, (len(layer_nodes), num_dst)))

        adjs = list(reversed(inner_to_outer))
        all_nodes = torch.tensor(layer_nodes, dtype=torch.long)
        return adjs, all_nodes
