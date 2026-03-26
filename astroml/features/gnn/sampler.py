from __future__ import annotations

from typing import List, Tuple

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
        fanout: List[int],
    ) -> None:
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.fanout = fanout

    def sample(
        self, target_nodes: torch.Tensor
    ) -> Tuple[List[Tuple[torch.Tensor, Tuple[int, int]]], torch.Tensor]:
        """Sample multi-hop neighborhood around target_nodes.

        Returns
        -------
        adjs : list of (edge_index, (num_src, num_dst))
            One entry per hop, ordered from outermost hop to innermost.
            edge_index uses local indices relative to the sampled subgraph.
        all_nodes : Tensor
            Union of all node IDs touched, ordered so that target_nodes
            appear at indices [0..len(target_nodes)-1].
        """
        adjs: List[Tuple[torch.Tensor, Tuple[int, int]]] = []
        current_nodes = target_nodes.clone()

        node_sets: List[torch.Tensor] = [current_nodes]

        for num_samples in reversed(self.fanout):
            src, dst = sample_neighbors(self.edge_index, current_nodes, num_samples)

            if src.numel() > 0:
                new_nodes = src[~torch.isin(src, current_nodes)]
                if new_nodes.numel() > 0:
                    current_nodes = torch.cat([current_nodes, new_nodes.unique()])
            node_sets.append(current_nodes.clone())

        # Build all_nodes: target first, then rest
        all_nodes = current_nodes
        non_target = all_nodes[~torch.isin(all_nodes, target_nodes)]
        all_nodes = torch.cat([target_nodes, non_target])

        # Build local-index adjacency lists
        global_to_local = {int(n): i for i, n in enumerate(all_nodes.tolist())}
        all_nodes_list = all_nodes.tolist()

        layer_dst_nodes = target_nodes
        for num_samples in self.fanout:
            src, dst = sample_neighbors(self.edge_index, layer_dst_nodes, num_samples)

            if src.numel() == 0:
                local_edge = torch.zeros((2, 0), dtype=torch.long)
                num_src = len(all_nodes_list)
                num_dst = len(layer_dst_nodes)
            else:
                # Ensure any newly sampled src nodes are in global_to_local
                for s in src.tolist():
                    if s not in global_to_local:
                        global_to_local[s] = len(all_nodes_list)
                        all_nodes_list.append(s)

                local_src = torch.tensor([global_to_local[int(s)] for s in src.tolist()], dtype=torch.long)
                local_dst = torch.tensor([global_to_local[int(d)] for d in dst.tolist()], dtype=torch.long)
                local_edge = torch.stack([local_src, local_dst])

                new_neighbors = src[~torch.isin(src, layer_dst_nodes)]
                layer_dst_nodes = torch.cat([layer_dst_nodes, new_neighbors.unique()])

                num_src = len(all_nodes_list)
                num_dst = local_edge[1].max().item() + 1 if local_edge.numel() > 0 else 0

            adjs.append((local_edge, (num_src, int(num_dst))))

        all_nodes = torch.tensor(all_nodes_list, dtype=torch.long)
        return adjs, all_nodes
