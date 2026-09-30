"""Tests for model security: adversarial robustness, extraction and poisoning.

Covers #645.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from numpy.typing import NDArray

from astroml.security.adversarial.attacks import (
    AttackConfig,
    AttackType,
    CarliniWagnerAttack,
    FGSMAttack,
    PGDAttack,
    generate_attack,
    numerical_gradient,
)
from astroml.security.adversarial.defenses import (
    AdversarialDetector,
    FeatureSqueezing,
    GaussianSmoothing,
    InputClipping,
    RobustnessEvaluator,
    adversarial_training_set,
)
from astroml.security.model_extraction import (
    ExtractionRisk,
    ExtractionSignal,
    ModelExtractionDetector,
)
from astroml.security.poisoning_detection import (
    PoisoningDetector,
    PoisoningType,
)
from astroml.security.scoring import SecurityTestPipeline

# A linear model: class 1 when the first two features sum above 1.0.
_WEIGHTS = np.array([4.0, 4.0])
_BIAS = -4.0


def linear_predict_proba(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return two-class probabilities from a fixed logistic model."""
    logits = np.asarray(x, dtype=np.float64) @ _WEIGHTS + _BIAS
    positive = 1.0 / (1.0 + np.exp(-logits))
    return np.column_stack([1.0 - positive, positive])


def linear_gradient(x: NDArray[np.float64], y: NDArray[np.int_]) -> NDArray[np.float64]:
    """Analytic gradient of cross-entropy loss w.r.t. the inputs."""
    probs = linear_predict_proba(x)[:, 1]
    # d/dx of -log p_y is (p1 - y) * w for the logistic model above.
    scale = (probs - np.asarray(y, dtype=np.float64))[:, None]
    return scale * _WEIGHTS[None, :]


@pytest.fixture
def dataset() -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """A small, confidently-classified evaluation set."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, size=(40, 2))
    y = linear_predict_proba(x).argmax(axis=1)
    return x, y


# ─── Attacks ─────────────────────────────────────────────────────────────────


class TestAttackConfig:
    """Configuration validation."""

    def test_defaults_derive_step_size(self) -> None:
        assert AttackConfig(epsilon=0.4).resolved_step_size == pytest.approx(0.1)

    def test_explicit_step_size_wins(self) -> None:
        assert AttackConfig(step_size=0.02).resolved_step_size == pytest.approx(0.02)

    @pytest.mark.parametrize(
        "kwargs",
        [{"epsilon": 0.0}, {"max_iterations": 0}, {"clip_range": (1.0, 0.0)}],
    )
    def test_invalid_config_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            AttackConfig(**kwargs)


class TestAttacks:
    """FGSM, PGD and Carlini-Wagner behaviour."""

    def test_fgsm_flips_predictions(self, dataset) -> None:
        x, y = dataset
        result = FGSMAttack(
            linear_predict_proba, linear_gradient, AttackConfig(epsilon=0.5)
        ).generate(x, y)
        assert result.attack is AttackType.FGSM
        assert result.success_rate > 0.0
        assert result.iterations == 1

    def test_fgsm_respects_epsilon_budget(self, dataset) -> None:
        x, y = dataset
        epsilon = 0.1
        result = FGSMAttack(
            linear_predict_proba, linear_gradient, AttackConfig(epsilon=epsilon)
        ).generate(x, y)
        delta = np.abs(result.adversarial_examples - x)
        assert delta.max() <= epsilon + 1e-9

    def test_attacks_stay_inside_clip_range(self, dataset) -> None:
        x, y = dataset
        result = PGDAttack(
            linear_predict_proba,
            linear_gradient,
            AttackConfig(epsilon=0.9, clip_range=(0.0, 1.0)),
        ).generate(x, y)
        assert result.adversarial_examples.min() >= 0.0
        assert result.adversarial_examples.max() <= 1.0

    def test_pgd_is_at_least_as_strong_as_fgsm(self, dataset) -> None:
        x, y = dataset
        config = AttackConfig(epsilon=0.15, max_iterations=25, random_start=False)
        fgsm = FGSMAttack(linear_predict_proba, linear_gradient, config).generate(x, y)
        pgd = PGDAttack(linear_predict_proba, linear_gradient, config).generate(x, y)
        assert pgd.success_rate >= fgsm.success_rate

    def test_pgd_is_reproducible_with_a_seed(self, dataset) -> None:
        x, y = dataset
        config = AttackConfig(epsilon=0.2, seed=42)
        first = PGDAttack(linear_predict_proba, linear_gradient, config).generate(x, y)
        second = PGDAttack(linear_predict_proba, linear_gradient, config).generate(x, y)
        np.testing.assert_allclose(first.adversarial_examples, second.adversarial_examples)

    def test_carlini_wagner_finds_smaller_distortions(self, dataset) -> None:
        x, y = dataset
        config = AttackConfig(epsilon=0.5, max_iterations=60, learning_rate=0.05)
        cw = CarliniWagnerAttack(linear_predict_proba, linear_gradient, config).generate(x, y)
        pgd = PGDAttack(linear_predict_proba, linear_gradient, config).generate(x, y)
        assert cw.mean_l2_distortion < pgd.mean_l2_distortion

    def test_carlini_wagner_rejects_bad_trade_off(self) -> None:
        with pytest.raises(ValueError):
            CarliniWagnerAttack(linear_predict_proba, trade_off=0.0)

    def test_targeted_attack_drives_towards_the_label(self, dataset) -> None:
        x, _ = dataset
        target = np.ones(x.shape[0], dtype=int)
        result = PGDAttack(
            linear_predict_proba,
            linear_gradient,
            AttackConfig(epsilon=0.9, targeted=True, max_iterations=40),
        ).generate(x, target)
        assert result.success_rate > 0.5

    def test_generate_attack_dispatches_by_name(self, dataset) -> None:
        x, y = dataset
        result = generate_attack("fgsm", linear_predict_proba, x, y, gradient=linear_gradient)
        assert result.attack is AttackType.FGSM

    def test_result_to_dict_is_serialisable(self, dataset) -> None:
        x, y = dataset
        payload = generate_attack(
            AttackType.FGSM, linear_predict_proba, x, y, gradient=linear_gradient
        ).to_dict()
        assert payload["sample_count"] == x.shape[0]
        assert 0.0 <= payload["success_rate"] <= 1.0

    def test_numerical_gradient_matches_analytic(self, dataset) -> None:
        x, y = dataset
        estimated = numerical_gradient(linear_predict_proba)(x, y)
        np.testing.assert_allclose(estimated, linear_gradient(x, y), atol=1e-4)

    def test_attack_works_without_an_analytic_gradient(self, dataset) -> None:
        x, y = dataset
        result = FGSMAttack(linear_predict_proba, config=AttackConfig(epsilon=0.5)).generate(x, y)
        assert result.success_rate > 0.0


# ─── Defences ────────────────────────────────────────────────────────────────


class TestDefenses:
    """Input-space defences and robustness evaluation."""

    def test_feature_squeezing_quantises(self) -> None:
        squeezer = FeatureSqueezing(bit_depth=1)
        squeezed = squeezer.transform(np.array([[0.1, 0.9]]))
        np.testing.assert_allclose(squeezed, [[0.0, 1.0]])

    def test_feature_squeezing_validates_settings(self) -> None:
        with pytest.raises(ValueError):
            FeatureSqueezing(bit_depth=0)
        with pytest.raises(ValueError):
            FeatureSqueezing(value_range=(1.0, 0.0))

    def test_feature_squeezing_removes_small_perturbations(self) -> None:
        clean = np.array([[0.25, 0.75]])
        perturbed = clean + 0.01
        squeezer = FeatureSqueezing(bit_depth=3)
        np.testing.assert_allclose(squeezer.transform(clean), squeezer.transform(perturbed))

    def test_gaussian_smoothing_averages_predictions(self, dataset) -> None:
        x, _ = dataset
        smoothed = GaussianSmoothing(sigma=0.01, samples=4).wrap(linear_predict_proba)
        probs = smoothed(x)
        assert probs.shape == (x.shape[0], 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    def test_gaussian_smoothing_validates_settings(self) -> None:
        with pytest.raises(ValueError):
            GaussianSmoothing(sigma=0.0)
        with pytest.raises(ValueError):
            GaussianSmoothing(samples=0)

    def test_input_clipping_fits_from_training_data(self) -> None:
        train = np.array([[0.0, 1.0], [1.0, 2.0], [0.5, 1.5]])
        clipper = InputClipping.from_training_data(train)
        clipped = clipper.transform(np.array([[-5.0, 99.0]]))
        np.testing.assert_allclose(clipped, [[0.0, 2.0]])

    def test_input_clipping_quantile_trimming(self) -> None:
        train = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
        clipper = InputClipping.from_training_data(train, quantile=0.2)
        assert clipper.transform(np.array([[100.0]]))[0, 0] < 100.0

    def test_input_clipping_rejects_bad_quantile(self) -> None:
        with pytest.raises(ValueError):
            InputClipping.from_training_data(np.zeros((3, 1)), quantile=0.5)

    def test_defense_wrap_applies_transform(self) -> None:
        defended = FeatureSqueezing(bit_depth=1).wrap(linear_predict_proba)
        np.testing.assert_allclose(
            defended(np.array([[0.1, 0.9]])),
            linear_predict_proba(np.array([[0.0, 1.0]])),
        )

    def test_detector_flags_adversarial_inputs(self, dataset) -> None:
        x, y = dataset
        adversarial = (
            PGDAttack(linear_predict_proba, linear_gradient, AttackConfig(epsilon=0.25))
            .generate(x, y)
            .adversarial_examples
        )

        detector = AdversarialDetector(linear_predict_proba, threshold=0.05)
        clean_flagged = detector.detect(x).flagged_count
        adversarial_flagged = detector.detect(adversarial).flagged_count
        assert adversarial_flagged >= clean_flagged

    def test_detector_calibration_targets_false_positive_rate(self, dataset) -> None:
        x, _ = dataset
        detector = AdversarialDetector(linear_predict_proba)
        threshold = detector.calibrate(x, false_positive_rate=0.1)
        assert threshold > 0
        assert detector.detect(x).flagged_count <= max(1, int(0.15 * x.shape[0]))

    def test_detector_validates_arguments(self, dataset) -> None:
        x, _ = dataset
        with pytest.raises(ValueError):
            AdversarialDetector(linear_predict_proba, threshold=0.0)
        with pytest.raises(ValueError):
            AdversarialDetector(linear_predict_proba).calibrate(x, false_positive_rate=1.0)

    def test_detection_result_to_dict(self, dataset) -> None:
        x, _ = dataset
        payload = AdversarialDetector(linear_predict_proba).detect(x).to_dict()
        assert payload["sample_count"] == x.shape[0]
        assert 0.0 <= payload["flagged_rate"] <= 1.0

    def test_adversarial_training_set_augments_data(self, dataset) -> None:
        x, y = dataset
        aug_x, aug_y = adversarial_training_set(
            linear_predict_proba, x, y, gradient=linear_gradient, ratio=0.5
        )
        assert aug_x.shape[0] == x.shape[0] + x.shape[0] // 2
        assert aug_y.shape[0] == aug_x.shape[0]
        # Adversarial rows keep their true labels — that is what makes it a defence.
        np.testing.assert_array_equal(aug_y[: x.shape[0]], y)

    def test_adversarial_training_set_validates_ratio(self, dataset) -> None:
        x, y = dataset
        with pytest.raises(ValueError):
            adversarial_training_set(linear_predict_proba, x, y, ratio=0.0)

    def test_robustness_evaluator_reports_per_attack(self, dataset) -> None:
        x, y = dataset
        report = RobustnessEvaluator(
            linear_predict_proba,
            gradient=linear_gradient,
            config=AttackConfig(epsilon=0.2),
        ).evaluate(x, y)

        assert report.clean_accuracy == pytest.approx(1.0)
        assert set(report.attack_results) == {"fgsm", "pgd"}
        assert 0.0 <= report.robustness_score <= 1.0
        assert report.robust_accuracy(AttackType.PGD) <= report.clean_accuracy
        assert "attacks" in report.to_dict()


# ─── Model extraction ────────────────────────────────────────────────────────


class TestModelExtractionDetector:
    """Behavioural extraction detection."""

    def test_insufficient_history_is_low_risk(self) -> None:
        detector = ModelExtractionDetector()
        detector.observe("client", np.array([0.5, 0.5]), 0.99)
        verdict = detector.assess("client")
        assert verdict.risk is ExtractionRisk.LOW
        assert verdict.details["reason"] == "insufficient history"

    def test_grid_probing_is_flagged(self) -> None:
        detector = ModelExtractionDetector(min_queries=10)
        start = datetime(2026, 5, 20, tzinfo=timezone.utc)
        grid = np.linspace(0.0, 1.0, 40)
        for index, value in enumerate(grid):
            detector.observe(
                "attacker",
                np.array([value, 1.0 - value]),
                confidence=0.5,
                timestamp=start + timedelta(seconds=index),
            )

        verdict = detector.assess("attacker", now=start + timedelta(seconds=40))
        assert ExtractionSignal.HIGH_VOLUME in verdict.signals
        assert ExtractionSignal.BOUNDARY_PROBING in verdict.signals
        assert verdict.risk in (ExtractionRisk.HIGH, ExtractionRisk.CRITICAL)

    def test_finite_difference_probing_is_flagged(self) -> None:
        detector = ModelExtractionDetector(min_queries=10)
        start = datetime(2026, 5, 20, tzinfo=timezone.utc)
        base = np.array([0.4, 0.4])
        for index in range(30):
            # Alternate between the base point and a single-feature nudge.
            point = base if index % 2 == 0 else base + np.array([1e-4, 0.0])
            detector.observe(
                "prober",
                point,
                confidence=0.95,
                timestamp=start + timedelta(seconds=index * 10),
            )

        verdict = detector.assess("prober", now=start + timedelta(minutes=5))
        assert ExtractionSignal.NEAR_DUPLICATE_PROBING in verdict.signals

    def test_normal_traffic_is_low_risk(self) -> None:
        detector = ModelExtractionDetector(min_queries=10)
        rng = np.random.default_rng(3)
        start = datetime(2026, 5, 20, tzinfo=timezone.utc)
        for index in range(30):
            detector.observe(
                "app",
                rng.normal(0.5, 0.05, size=2),
                confidence=0.97,
                timestamp=start + timedelta(seconds=index * 15),
            )

        verdict = detector.assess("app", now=start + timedelta(minutes=8))
        assert verdict.risk is ExtractionRisk.LOW
        assert verdict.signals == ()

    def test_stale_queries_fall_out_of_the_window(self) -> None:
        detector = ModelExtractionDetector(min_queries=2, window=timedelta(minutes=1))
        old = datetime(2026, 5, 20, tzinfo=timezone.utc)
        for index in range(10):
            detector.observe("client", np.array([0.1, 0.2]), 0.9, timestamp=old)
        verdict = detector.assess("client", now=old + timedelta(hours=1))
        assert verdict.query_count == 0

    def test_confidence_bounds_are_validated(self) -> None:
        detector = ModelExtractionDetector()
        with pytest.raises(ValueError):
            detector.observe("client", np.array([0.5]), 1.5)

    def test_constructor_validates_arguments(self) -> None:
        with pytest.raises(ValueError):
            ModelExtractionDetector(window=timedelta(0))
        with pytest.raises(ValueError):
            ModelExtractionDetector(min_queries=1)

    def test_reset_clears_history(self) -> None:
        detector = ModelExtractionDetector()
        detector.observe("a", np.array([0.1]), 0.5)
        detector.observe("b", np.array([0.1]), 0.5)
        detector.reset("a")
        assert detector.clients() == ["b"]
        detector.reset()
        assert detector.clients() == []

    def test_report_ranks_clients(self) -> None:
        detector = ModelExtractionDetector(min_queries=5)
        start = datetime(2026, 5, 20, tzinfo=timezone.utc)
        for index in range(20):
            detector.observe(
                "attacker",
                np.array([index / 20.0, 1 - index / 20.0]),
                0.5,
                timestamp=start + timedelta(seconds=index),
            )
            detector.observe(
                "app",
                np.array([0.5, 0.5]),
                0.99,
                timestamp=start + timedelta(seconds=index * 30),
            )

        report = detector.report(now=start + timedelta(minutes=9))
        assert report["client_count"] == 2
        scores = [verdict["score"] for verdict in report["verdicts"]]
        assert scores == sorted(scores, reverse=True)


# ─── Poisoning detection ─────────────────────────────────────────────────────


class TestPoisoningDetector:
    """Training-data poisoning screening."""

    @pytest.fixture
    def clean_data(self) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        rng = np.random.default_rng(11)
        class_a = rng.normal(0.0, 0.05, size=(30, 3))
        class_b = rng.normal(3.0, 0.05, size=(30, 3))
        x = np.vstack([class_a, class_b])
        y = np.array([0] * 30 + [1] * 30)
        return x, y

    def test_clean_data_passes(self, clean_data) -> None:
        x, y = clean_data
        report = PoisoningDetector().detect(x, y)
        assert report.is_clean
        assert report.contamination_rate == 0.0

    def test_label_flips_are_detected(self, clean_data) -> None:
        x, y = clean_data
        poisoned_y = y.copy()
        poisoned_y[0] = 1  # a class-0 point labelled class 1
        report = PoisoningDetector().detect(x, poisoned_y)

        finding = next(f for f in report.findings if f.poisoning_type is PoisoningType.LABEL_FLIP)
        assert 0 in finding.suspicious_indices

    def test_injected_outliers_are_detected(self, clean_data) -> None:
        x, y = clean_data
        poisoned_x = np.vstack([x, np.full((1, 3), 50.0)])
        poisoned_y = np.concatenate([y, [1]])
        report = PoisoningDetector().detect(poisoned_x, poisoned_y)

        finding = next(
            f for f in report.findings if f.poisoning_type is PoisoningType.OUTLIER_INJECTION
        )
        assert poisoned_x.shape[0] - 1 in finding.suspicious_indices

    def test_backdoor_trigger_is_detected(self, clean_data) -> None:
        x, y = clean_data
        # Five rows carrying an identical rare marker, all labelled class 1.
        trigger = np.tile(np.array([[0.0, 0.0, 42.0]]), (5, 1))
        poisoned_x = np.vstack([x, trigger])
        poisoned_y = np.concatenate([y, np.ones(5, dtype=int)])

        finding = next(
            f
            for f in PoisoningDetector().detect(poisoned_x, poisoned_y).findings
            if f.poisoning_type is PoisoningType.BACKDOOR_TRIGGER
        )
        assert finding.suspicious_count >= 5
        assert finding.details["triggers"]

    def test_sanitize_removes_flagged_rows(self, clean_data) -> None:
        x, y = clean_data
        poisoned_x = np.vstack([x, np.full((1, 3), 50.0)])
        poisoned_y = np.concatenate([y, [1]])
        clean_x, clean_y, report = PoisoningDetector().sanitize(poisoned_x, poisoned_y)

        assert clean_x.shape[0] == poisoned_x.shape[0] - len(report.suspicious_indices)
        assert clean_y.shape[0] == clean_x.shape[0]

    def test_report_serialises(self, clean_data) -> None:
        x, y = clean_data
        payload = PoisoningDetector().detect(x, y).to_dict()
        assert payload["sample_count"] == x.shape[0]
        assert len(payload["findings"]) == 3

    def test_shape_mismatch_rejected(self, clean_data) -> None:
        x, y = clean_data
        with pytest.raises(ValueError, match="same number of samples"):
            PoisoningDetector().detect(x, y[:-1])

    def test_one_dimensional_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D array"):
            PoisoningDetector().detect(np.zeros(5), np.zeros(5, dtype=int))

    def test_constructor_validates_arguments(self) -> None:
        with pytest.raises(ValueError):
            PoisoningDetector(n_neighbors=0)
        with pytest.raises(ValueError):
            PoisoningDetector(label_flip_threshold=0.0)
        with pytest.raises(ValueError):
            PoisoningDetector(outlier_z_threshold=0.0)

    def test_subsampling_keeps_large_sets_tractable(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(0.0, 1.0, size=(200, 2))
        y = (x[:, 0] > 0).astype(int)
        detector = PoisoningDetector(max_pairwise_samples=50)
        assert detector.detect(x, y).sample_count == 200


# ─── Security scoring ────────────────────────────────────────────────────────


class TestSecurityTestPipeline:
    """Composite scoring and report generation."""

    def test_full_scan_produces_a_graded_score(self, dataset) -> None:
        x, y = dataset
        pipeline = SecurityTestPipeline(
            "fraud-gnn",
            linear_predict_proba,
            gradient=linear_gradient,
            attack_config=AttackConfig(epsilon=0.05),
        )
        result = pipeline.run(x_eval=x, y_eval=y, x_train=x, y_train=y)

        assert result.model_name == "fraud-gnn"
        assert 0.0 <= result.score.overall <= 100.0
        assert result.score.grade in {"A", "B", "C", "D", "F"}
        assert set(result.score.components) == {"robustness", "poisoning"}

    def test_partial_scan_renormalises_the_score(self, dataset) -> None:
        x, y = dataset
        result = SecurityTestPipeline("m", linear_predict_proba, gradient=linear_gradient).run(
            x_train=x, y_train=y
        )
        assert set(result.score.components) == {"poisoning"}
        assert result.robustness is None

    def test_scan_with_no_data_scores_nothing(self) -> None:
        result = SecurityTestPipeline("m", linear_predict_proba).run()
        assert result.score.grade == "N/A"
        assert result.score.overall == 0.0

    def test_extraction_verdicts_feed_the_score(self, dataset) -> None:
        x, y = dataset
        detector = ModelExtractionDetector(min_queries=5)
        start = datetime(2026, 5, 20, tzinfo=timezone.utc)
        for index in range(20):
            detector.observe(
                "attacker",
                np.array([index / 20.0, 1 - index / 20.0]),
                0.5,
                timestamp=start + timedelta(seconds=index),
            )

        result = SecurityTestPipeline(
            "m",
            linear_predict_proba,
            gradient=linear_gradient,
            extraction_detector=detector,
        ).run(x_eval=x, y_eval=y)
        assert "extraction" in result.score.components

    def test_markdown_report_covers_every_section(self, dataset) -> None:
        x, y = dataset
        result = SecurityTestPipeline(
            "fraud-gnn", linear_predict_proba, gradient=linear_gradient
        ).run(x_eval=x, y_eval=y, x_train=x, y_train=y)

        markdown = result.to_markdown()
        assert "# Model Security Report — fraud-gnn" in markdown
        assert "## Adversarial robustness" in markdown
        assert "## Training data poisoning" in markdown

    def test_result_is_serialisable(self, dataset) -> None:
        x, y = dataset
        payload = (
            SecurityTestPipeline("m", linear_predict_proba, gradient=linear_gradient)
            .run(x_eval=x, y_eval=y)
            .to_dict()
        )
        assert payload["model_name"] == "m"
        assert payload["poisoning"] is None

    def test_empty_model_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            SecurityTestPipeline("", linear_predict_proba)
