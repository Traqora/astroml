"""Cold-start scoring tests for InductiveAnomalyScorer (#900).

A cold-start account is one with very little or no history in the window
being scored. These tests pin down what the scorer returns for those cases.
"""

from __future__ import annotations

import math

import pytest
import torch

from astroml.models.deep_svdd import DeepSVDD
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.pipeline.inductive import InductiveGraphSAGE
from astroml.pipeline.scoring import InductiveAnomalyScorer


def _scorer(fanout=(3, 2)) -> InductiveAnomalyScorer:
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=16, output_dim=8, num_layers=len(fanout), dropout=0.0
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=list(fanout), device="cpu")
    svdd = DeepSVDD(input_dim=8, hidden_dims=[16, 8], dropout=0.0, device="cpu")
    svdd.center = torch.zeros(8)
    return InductiveAnomalyScorer(pipeline, svdd)


def _history():
    return [
        {"src": "A", "dst": "B", "amount": 100.0, "timestamp": 1000.0, "asset": "XLM"},
        {"src": "B", "dst": "C", "amount": 50.0, "timestamp": 2000.0, "asset": "XLM"},
        {"src": "C", "dst": "A", "amount": 25.0, "timestamp": 3000.0, "asset": "USD"},
    ]


@pytest.mark.parametrize("seed", range(20))
def test_account_with_single_edge_gets_finite_score(seed):
    torch.manual_seed(seed)
    edges = _history() + [
        {"src": "A", "dst": "NEW", "amount": 5.0, "timestamp": 3900.0, "asset": "XLM"}
    ]
    scores = _scorer().score_new_accounts(edges, ["NEW"], ref_time=4000.0)
    assert math.isfinite(scores["NEW"])
    assert scores["NEW"] >= 0.0


def test_account_that_only_sends_gets_finite_score():
    torch.manual_seed(0)
    edges = _history() + [
        {"src": "NEW", "dst": "A", "amount": 5.0, "timestamp": 3900.0, "asset": "XLM"}
    ]
    scores = _scorer().score_new_accounts(edges, ["NEW"], ref_time=4000.0)
    assert math.isfinite(scores["NEW"])


def test_unseen_account_is_scored_as_inf_alongside_known_ones():
    torch.manual_seed(0)
    scores = _scorer().score_new_accounts(_history(), ["A", "GHOST"], ref_time=4000.0)
    assert math.isfinite(scores["A"])
    assert scores["GHOST"] == float("inf")


def test_every_requested_account_gets_a_score():
    torch.manual_seed(0)
    accounts = ["A", "B", "C", "GHOST"]
    scores = _scorer().score_new_accounts(_history(), accounts, ref_time=4000.0)
    assert set(scores) == set(accounts)


def test_empty_edge_list_does_not_crash():
    torch.manual_seed(0)
    scores = _scorer().score_new_accounts([], ["NEW"], ref_time=4000.0)
    assert set(scores) == {"NEW"}
