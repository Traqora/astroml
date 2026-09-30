"""Tests for automated data labeling and annotation pipeline (#624)."""

from __future__ import annotations

import numpy as np
import pytest

from astroml.preprocessing.labeling import (
    ActiveLearner,
    BatchLabelingStrategy,
    ConflictResolver,
    HybridStrategy,
    LabelingFunction,
    LabelingPipeline,
    ReviewQueue,
    UncertaintySampling,
    WeakSupervisionModel,
    create_binary_lfs,
)
from astroml.preprocessing.labeling.active_learning import (
    EntropySampling,
    MarginSampling,
)
from astroml.preprocessing.labeling.review_queue import LabelQualityMetrics
from astroml.preprocessing.labeling.weak_supervision import ABSTAIN, MajorityVoter

# ---------------------------------------------------------------------------
# Active learning
# ---------------------------------------------------------------------------


class TestActiveLearning:
    def test_uncertainty_sampling_scores(self) -> None:
        pool = list(range(20))
        # 3-class probabilities: model is confident about first 10, uncertain about rest
        probs = np.zeros((20, 3))
        probs[:10, 0] = 0.9
        probs[:10, 1] = 0.05
        probs[:10, 2] = 0.05
        probs[10:, 0] = 0.34
        probs[10:, 1] = 0.33
        probs[10:, 2] = 0.33

        strategy = UncertaintySampling()
        scores = strategy.score_samples(pool, model_probs=probs)
        assert len(scores) == 20
        # Uncertain samples (10..19) should score higher
        mean_uncertain = np.mean([s.score for s in scores[10:]])
        mean_confident = np.mean([s.score for s in scores[:10]])
        assert mean_uncertain > mean_confident

    def test_margin_sampling(self) -> None:
        pool = list(range(10))
        probs = np.array([
            [0.9, 0.05, 0.05],  # high margin
            [0.4, 0.35, 0.25],  # low margin
            [0.5, 0.25, 0.25],  # medium margin
            [0.8, 0.1, 0.1],
            [0.35, 0.33, 0.32],
        ])
        strategy = MarginSampling()
        scores = strategy.score_samples(pool, model_probs=probs)
        # Index 4 (lowest margin) should have highest score
        best = max(scores, key=lambda s: s.score)
        assert best.index == 4

    def test_entropy_sampling(self) -> None:
        pool = list(range(5))
        # Uniform distribution (max entropy) vs peaked
        probs = np.array([
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            [0.98, 0.01, 0.01],
        ])
        strategy = EntropySampling()
        scores = strategy.score_samples(pool, model_probs=probs)
        assert scores[0].score > scores[1].score

    def test_active_learner_rounds(self) -> None:
        pool = list(range(50))
        strategy = UncertaintySampling()
        learner = ActiveLearner(strategy, pool, batch_size=5, max_rounds=3)

        def model(samples: list[int]) -> np.ndarray:
            rng = np.random.default_rng(42)
            return rng.dirichlet(np.ones(3), size=len(samples))

        results = learner.run(model)
        assert len(results) == 3
        assert learner.labeled_count == 15  # 3 rounds * 5 batch
        assert learner.unlabeled_count == 35

    def test_hybrid_strategy(self) -> None:
        pool = list(range(10))
        probs = np.full((10, 3), 1.0 / 3)

        uncertainty = UncertaintySampling()
        diversity = UncertaintySampling()  # reusing uncertainty for simplicity
        hybrid = HybridStrategy(uncertainty, diversity, alpha=0.7)

        scores = hybrid.score_samples(pool, model_probs=probs)
        assert len(scores) == 10
        for s in scores:
            assert 0 <= s.uncertainty <= 1
            assert 0 <= s.diversity <= 1


# ---------------------------------------------------------------------------
# Weak supervision
# ---------------------------------------------------------------------------


class TestWeakSupervision:
    def test_lf_abstain(self) -> None:
        lf = LabelingFunction(
            name="test",
            fn=lambda x: 1 if isinstance(x, int) and x > 5 else ABSTAIN,
        )
        assert lf.apply(7) == 1
        assert lf.apply(3) == ABSTAIN

    def test_lf_error_abstains(self) -> None:
        lf = LabelingFunction(name="broken", fn=lambda x: 1 / 0)  # type: ignore[arg-type]
        assert lf.apply(1) == ABSTAIN

    def test_weak_supervision_model(self) -> None:
        lfs = create_binary_lfs()
        samples = ["hello world", "hi", "this is a longer sentence"]
        model = WeakSupervisionModel(lfs)
        model.fit(samples)
        probs = model.predict_proba(samples)
        assert probs.shape == (3, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        labels = model.predict(samples)
        assert labels.shape == (3,)

    def test_majority_voter(self) -> None:
        lfs = [
            LabelingFunction(name="lf1", fn=lambda _: 0),
            LabelingFunction(name="lf2", fn=lambda _: 1),
            LabelingFunction(name="lf3", fn=lambda _: 1),
        ]
        voter = MajorityVoter(lfs)
        samples = [None]
        voter.fit(samples)
        probs = voter.predict_proba(samples)
        # 1 vote for 0, 2 votes for 1
        assert probs[0, 1] > probs[0, 0]

    def test_lf_coverage(self) -> None:
        lf = LabelingFunction(name="half", fn=lambda x: x if x % 2 == 0 else ABSTAIN)
        model = WeakSupervisionModel([lf])
        samples = list(range(100))
        model.fit(samples)
        assert abs(lf.coverage - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


class TestReviewQueue:
    def test_enqueue_and_dequeue(self) -> None:
        queue = ReviewQueue(max_queue_size=100)
        queue.enqueue("item1", {"data": 1}, {"lf1": 0, "lf2": 1}, conflict_score=0.5)
        queue.enqueue("item2", {"data": 2}, {"lf1": 1, "lf2": 1}, conflict_score=0.1)
        queue.enqueue("item3", {"data": 3}, {"lf1": 0, "lf2": 0}, conflict_score=0.9)

        batch = queue.dequeue_batch(2)
        assert len(batch) == 2
        # Highest priority first
        assert batch[0].item_id == "item3"
        assert batch[1].item_id == "item1"

    def test_resolve(self) -> None:
        queue = ReviewQueue()
        queue.enqueue("item1", None, {"v": 1})
        assert queue.resolve("item1", "approved")
        assert not queue.resolve("nonexistent", "nope")

    def test_quality_metrics(self) -> None:
        metrics = LabelQualityMetrics(labeler_id="lf1")
        metrics.update(total=10, agreed=8, avg_conf=0.9, rejected=1)
        assert metrics.total_labels == 10
        assert metrics.agreement_with_consensus == 0.8
        assert pytest.approx(metrics.avg_confidence, 0.01) == 0.9
        assert metrics.review_rejection_rate == 0.1

    def test_conflict_resolver_majority(self) -> None:
        resolver = ConflictResolver(strategy="majority")
        winner, conf = resolver.resolve([0, 0, 0, 1])
        assert winner == 0
        assert conf == 0.75

    def test_conflict_resolver_weighted(self) -> None:
        resolver = ConflictResolver(strategy="weighted")
        votes = [0, 1]
        weights = [0.1, 0.9]
        winner, conf = resolver.resolve(votes, weights=weights)
        assert winner == 1


# ---------------------------------------------------------------------------
# Labeling pipeline
# ---------------------------------------------------------------------------


class TestLabelingPipeline:
    def test_pipeline_run(self) -> None:
        lfs = create_binary_lfs()
        strategy = BatchLabelingStrategy(lfs)
        pipeline = LabelingPipeline(strategy, auto_accept_threshold=0.0)

        samples = ["short", "a long enough sentence for testing", "hi"]
        result = pipeline.run(samples)
        assert result.labels.shape == (3,)
        assert result.probabilities is not None
        assert result.probabilities.shape[1] == 2

    def test_pipeline_dashboard(self) -> None:
        lfs = create_binary_lfs()
        strategy = BatchLabelingStrategy(lfs)
        pipeline = LabelingPipeline(strategy)
        pipeline.run(["a", "b", "c"])
        dashboard = pipeline.get_dashboard()
        assert "pipeline_stats" in dashboard
        assert "quality_report" in dashboard
        assert "strategy_stats" in dashboard