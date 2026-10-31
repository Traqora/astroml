"""Regression test for docs/GRAPHSAGE_INDUCTIVE_LEARNING.md (GitHub Issue #843).

Verifies that every module/class/function the doc describes as the usage
surface for inductive GraphSAGE actually exists and is importable, so a
rename or removal breaks this test instead of silently leaving the doc
describing an API that no longer exists.
"""

from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).parent.parent
_DOC = _ROOT / "docs" / "GRAPHSAGE_INDUCTIVE_LEARNING.md"


def test_doc_exists() -> None:
    assert _DOC.exists(), "docs/GRAPHSAGE_INDUCTIVE_LEARNING.md is missing"


def test_sampler_api_matches_doc() -> None:
    from astroml.features.gnn.sampler import MultiHopSampler

    assert hasattr(MultiHopSampler, "sample")


def test_encoder_api_matches_doc() -> None:
    from astroml.models.sage_encoder import InductiveSAGEEncoder

    assert InductiveSAGEEncoder is not None


def test_training_api_matches_doc() -> None:
    from astroml.training.train_sage import build_reconstruction_target, train_epoch

    assert callable(build_reconstruction_target)
    assert callable(train_epoch)


def test_pipeline_api_matches_doc() -> None:
    from astroml.pipeline.inductive import InductiveGraphSAGE

    assert hasattr(InductiveGraphSAGE, "embed_nodes")
    assert hasattr(InductiveGraphSAGE, "embed_snapshot")


def test_scoring_api_matches_doc() -> None:
    from astroml.pipeline.scoring import InductiveAnomalyScorer

    assert hasattr(InductiveAnomalyScorer, "score_new_accounts")


def test_deep_svdd_importable() -> None:
    from astroml.models.deep_svdd import DeepSVDD

    assert DeepSVDD is not None
