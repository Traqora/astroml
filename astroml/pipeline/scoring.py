from __future__ import annotations

from typing import Dict, List

import torch

from astroml.pipeline.inductive import InductiveGraphSAGE, _FEATURE_COLS
from astroml.features.node_features import compute_node_features
from astroml.models.deep_svdd import DeepSVDD
from ..cache import cached_prediction



class InductiveAnomalyScorer:
    """Connects inductive GraphSAGE embeddings to Deep SVDD for anomaly scoring.

    Parameters
    ----------
    pipeline : InductiveGraphSAGE
        Embedding pipeline for producing node representations.
    svdd : DeepSVDD
        Trained Deep SVDD model for anomaly scoring.
    mode : str
        'embeddings_only' feeds GraphSAGE embeddings to SVDD directly.
        'concatenated' concatenates embeddings with raw node features.
    """

    def __init__(
        self,
        pipeline: InductiveGraphSAGE,
        svdd: DeepSVDD,
        mode: str = 'embeddings_only',
    ) -> None:
        assert mode in ('embeddings_only', 'concatenated')
        self.pipeline = pipeline
        self.svdd = svdd
        self.mode = mode

    @cached_prediction(ttl_seconds=300)
    def score_new_accounts(
        self,
        edges: List[Dict],
        account_ids: List[str],
        ref_time: float,
    ) -> Dict[str, float]:
        """Produce anomaly scores for the given accounts.

        Parameters
        ----------
        edges : list of dict
            Transaction edges for graph context.
        account_ids : list of str
            Accounts to score.
        ref_time : float
            Reference timestamp for feature computation.

        Returns
        -------
        dict mapping account_id -> anomaly score (float).
            Higher scores indicate more anomalous.
        """
        embeddings = self.pipeline.embed_nodes(edges, account_ids, ref_time)

        if self.mode == 'concatenated':
            feat_df = compute_node_features(edges, ref_time=ref_time)

        results: Dict[str, float] = {}
        for node_id in account_ids:
            if node_id not in embeddings:
                results[node_id] = float('inf')
                continue

            emb = embeddings[node_id]

            if self.mode == 'concatenated':
                if node_id in feat_df.index:
                    raw = torch.tensor(
                        feat_df.loc[node_id][_FEATURE_COLS].values,
                        dtype=torch.float32,
                    )
                else:
                    raw = torch.zeros(len(_FEATURE_COLS))
                svdd_input = torch.cat([emb, raw]).unsqueeze(0)
            else:
                svdd_input = emb.unsqueeze(0)

            score = self.svdd.predict(svdd_input)
            results[node_id] = float(score[0])

        return results
