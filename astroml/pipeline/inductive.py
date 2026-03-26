from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from astroml.features.node_features import compute_node_features
from astroml.features.graph.snapshot import window_snapshot
from astroml.features.gnn.sampler import MultiHopSampler
from astroml.models.sage_encoder import InductiveSAGEEncoder


# Column order from compute_node_features output
_FEATURE_COLS = [
    'in_degree', 'out_degree', 'total_received', 'total_sent',
    'account_age', 'first_seen', 'unique_asset_count', 'asset_entropy',
]


class InductiveGraphSAGE:
    """Orchestrator for inductive node embedding via GraphSAGE.

    Ties together graph snapshots, feature computation, neighbor sampling,
    and the multi-layer SAGEConv encoder into a single callable.

    Parameters
    ----------
    encoder : InductiveSAGEEncoder
        Trained (or untrained) multi-layer encoder.
    fanout : list[int]
        Neighbors per hop, passed to MultiHopSampler.
    device : str
        Torch device for computation.
    """

    def __init__(
        self,
        encoder: InductiveSAGEEncoder,
        fanout: List[int],
        device: str = 'cpu',
    ) -> None:
        self.encoder = encoder.to(device)
        self.fanout = fanout
        self.device = device

    def embed_nodes(
        self,
        edges: List[Dict],
        target_nodes: List[str],
        ref_time: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute embeddings for target_nodes given a set of edges.

        Parameters
        ----------
        edges : list of dict
            Transaction edges with keys: src, dst, amount, timestamp, asset.
        target_nodes : list of str
            Account IDs to produce embeddings for.
        ref_time : float
            Reference timestamp for feature computation.

        Returns
        -------
        dict mapping node_id -> Tensor of shape [output_dim]
        """
        # 1. Compute node features
        feat_df = compute_node_features(edges, ref_time=ref_time)
        all_node_ids = list(feat_df.index)
        node_to_idx = {n: i for i, n in enumerate(all_node_ids)}

        # 2. Build edge_index in integer space
        src_indices = []
        dst_indices = []
        for e in edges:
            s, d = e.get('src'), e.get('dst')
            if s in node_to_idx and d in node_to_idx:
                src_indices.append(node_to_idx[s])
                dst_indices.append(node_to_idx[d])

        if not src_indices:
            out_dim = self.encoder.convs[-1].out_dim
            return {n: torch.zeros(out_dim) for n in target_nodes}

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

        # 3. Convert target_nodes to integer indices
        target_idx = torch.tensor(
            [node_to_idx[n] for n in target_nodes if n in node_to_idx],
            dtype=torch.long,
        )
        valid_targets = [n for n in target_nodes if n in node_to_idx]

        if target_idx.numel() == 0:
            out_dim = self.encoder.convs[-1].out_dim
            return {n: torch.zeros(out_dim) for n in target_nodes}

        # 4. Sample neighborhoods
        sampler = MultiHopSampler(edge_index, len(all_node_ids), self.fanout)
        adjs, sampled_nodes = sampler.sample(target_idx)

        # 5. Slice feature matrix to sampled nodes
        feat_matrix = feat_df.loc[
            [all_node_ids[i] for i in sampled_nodes.tolist()]
        ][_FEATURE_COLS].values
        x = torch.tensor(feat_matrix, dtype=torch.float32).to(self.device)

        # 6. Forward through encoder
        self.encoder.train(False)
        with torch.no_grad():
            embeddings = self.encoder(x, adjs)

        # 7. Map back to node IDs
        result = {}
        for i, node_id in enumerate(valid_targets):
            result[node_id] = embeddings[i].cpu()

        return result

    def embed_snapshot(
        self,
        edges: Sequence,
        start_ts: int,
        end_ts: int,
        target_nodes: List[str],
        ref_time: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute embeddings using a time-windowed snapshot.

        Parameters
        ----------
        edges : sequence of snapshot Edge objects
            Full edge list (will be filtered by time window).
        start_ts : int
            Start of time window (inclusive).
        end_ts : int
            End of time window (inclusive).
        target_nodes : list of str
            Account IDs to embed.
        ref_time : float, optional
            Reference time for features. Defaults to end_ts.

        Returns
        -------
        dict mapping node_id -> Tensor of shape [output_dim]
        """
        _, window_edges = window_snapshot(edges, start_ts, end_ts)

        edge_dicts = [
            {'src': e.src, 'dst': e.dst, 'amount': 0.0, 'timestamp': float(e.timestamp)}
            for e in window_edges
        ]

        return self.embed_nodes(
            edge_dicts,
            target_nodes,
            ref_time=ref_time if ref_time is not None else float(end_ts),
        )
