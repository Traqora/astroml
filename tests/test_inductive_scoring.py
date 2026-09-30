from __future__ import annotations

import numpy as np
import torch

from astroml.models.deep_svdd import DeepSVDD
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.pipeline.inductive import InductiveGraphSAGE
from astroml.pipeline.scoring import InductiveAnomalyScorer


def _make_edges():
    return [
        {"src": "A", "dst": "B", "amount": 100.0, "timestamp": 1000.0, "asset": "XLM"},
        {"src": "B", "dst": "C", "amount": 50.0, "timestamp": 2000.0, "asset": "XLM"},
        {"src": "C", "dst": "D", "amount": 25.0, "timestamp": 3000.0, "asset": "USD"},
        {"src": "A", "dst": "D", "amount": 10.0, "timestamp": 3500.0, "asset": "XLM"},
    ]


def test_score_new_accounts():
    """Scorer returns finite anomaly scores for new accounts."""
    encoder = InductiveSAGEEncoder(
        input_dim=8,
        hidden_dim=16,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        aggregator="mean",
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[3, 2], device="cpu")

    svdd = DeepSVDD(input_dim=8, hidden_dims=[16, 8], dropout=0.0, device="cpu")
    svdd.center = torch.zeros(8)

    scorer = InductiveAnomalyScorer(pipeline, svdd)
    scores = scorer.score_new_accounts(_make_edges(), ["C", "D"], ref_time=4000.0)

    assert "C" in scores
    assert "D" in scores
    assert np.isfinite(scores["C"])
    assert np.isfinite(scores["D"])


def test_score_concatenated_mode():
    """Concatenated mode produces scores using embeddings + raw features."""
    encoder = InductiveSAGEEncoder(
        input_dim=8,
        hidden_dim=8,
        output_dim=4,
        num_layers=1,
        dropout=0.0,
        aggregator="mean",
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[3], device="cpu")

    # input_dim = encoder output (4) + raw features (8) = 12
    svdd = DeepSVDD(input_dim=12, hidden_dims=[8, 4], dropout=0.0, device="cpu")
    svdd.center = torch.zeros(4)

    scorer = InductiveAnomalyScorer(pipeline, svdd, mode="concatenated")
    scores = scorer.score_new_accounts(_make_edges(), ["B"], ref_time=4000.0)

    assert "B" in scores
    assert np.isfinite(scores["B"])
