"""Regression tests for anomaly detection and model calibration (issue #717).

Pins expected outputs so a subtle behavioural change is caught rather than
absorbed. Two paths are covered:

* **Anomaly detection** — a statistical detector run over synthetic
  injected-fraud data produced by ``inject_synthetic_fraud``. The injector is
  seeded, so the fraud it plants is fixed and the detector's verdicts on it can
  be asserted exactly.
* **Calibration** — Platt and isotonic calibrators, whose contract (monotonicity,
  bounds, idempotent refit) is exactly the sort of thing that degrades quietly.

Both use fixed seeds; a failure here means behaviour changed, not that a random
draw went the other way.
"""

from __future__ import annotations

import numpy as np
import pytest

from astroml.ingestion.synthetic_fraud_injector import (
    SybilConfig,
    WashLoopConfig,
    inject_synthetic_fraud,
)
from astroml.training.calibration.isotonic import IsotonicCalibrator
from astroml.training.calibration.platt import PlattCalibrator

SEED = 7


def _baseline_transactions(n: int = 40) -> list[dict]:
    """Deterministic, fraud-free ledger to inject into."""
    rng = np.random.default_rng(SEED)
    amounts = rng.normal(loc=100.0, scale=10.0, size=n)
    return [
        {
            "id": i,
            "source_account": f"account_{i % 8}",
            "destination_account": f"account_{(i + 3) % 8}",
            "amount": float(round(amount, 4)),
            "created_at": f"2026-01-01T00:{i:02d}:00+00:00",
        }
        for i, amount in enumerate(amounts, start=1)
    ]


class TestSyntheticFraudInjection:
    """The fixture the detector is judged against must itself be stable."""

    def test_injection_is_deterministic_for_a_fixed_seed(self):
        base = _baseline_transactions()

        first, summary_a = inject_synthetic_fraud(base, seed=SEED)
        second, summary_b = inject_synthetic_fraud(base, seed=SEED)

        assert summary_a == summary_b
        assert first == second

    def test_injection_counts_are_pinned(self):
        base = _baseline_transactions()

        augmented, summary = inject_synthetic_fraud(
            base,
            seed=SEED,
            sybil=SybilConfig(clusters=2, cluster_size=5, tx_per_member=3),
            wash=WashLoopConfig(),
        )

        # 2 clusters x 5 members x 3 transactions.
        assert summary.sybil_transactions == 30
        assert summary.wash_loop_transactions > 0
        assert summary.injected_transactions == (
            summary.sybil_transactions + summary.wash_loop_transactions
        )
        assert len(augmented) == len(base) + summary.injected_transactions
        assert summary.original_transactions == len(base)
        assert summary.total_transactions == len(augmented)

    def test_injection_leaves_the_original_transactions_untouched(self):
        base = _baseline_transactions()

        augmented, _ = inject_synthetic_fraud(base, seed=SEED)

        assert augmented[: len(base)] == base

    def test_a_different_seed_changes_the_injected_amounts(self):
        base = _baseline_transactions()

        a, _ = inject_synthetic_fraud(base, seed=SEED)
        b, _ = inject_synthetic_fraud(base, seed=SEED + 1)

        assert a != b


def _zscore_anomaly_scores(transactions: list[dict]) -> np.ndarray:
    """Per-account transaction-count z-score.

    A deliberately simple, dependency-free detector: sybil and wash patterns
    both show up as accounts transacting far more often than the population.
    Pinning *this* keeps the regression test about the injected data and the
    scoring contract rather than about a particular model's weights.
    """
    counts: dict[str, int] = {}
    for tx in transactions:
        counts[tx["source_account"]] = counts.get(tx["source_account"], 0) + 1

    accounts = sorted(counts)
    values = np.array([counts[a] for a in accounts], dtype=float)
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


class TestAnomalyDetectionOnInjectedFraud:
    def test_injected_fraud_raises_the_maximum_anomaly_score(self):
        base = _baseline_transactions()
        clean_scores = _zscore_anomaly_scores(base)

        augmented, _ = inject_synthetic_fraud(base, seed=SEED)
        fraud_scores = _zscore_anomaly_scores(augmented)

        assert fraud_scores.max() > clean_scores.max()

    def test_the_flagged_accounts_are_the_injected_ones(self):
        base = _baseline_transactions()
        augmented, _ = inject_synthetic_fraud(base, seed=SEED)

        counts: dict[str, int] = {}
        for tx in augmented:
            counts[tx["source_account"]] = counts.get(tx["source_account"], 0) + 1

        accounts = sorted(counts)
        scores = _zscore_anomaly_scores(augmented)
        flagged = {a for a, s in zip(accounts, scores) if s > 2.0}

        assert flagged, "the detector must flag something on injected fraud"
        # Every flagged account is synthetic, not an original participant.
        assert all(
            a.startswith(("sybil", "wash")) for a in flagged
        ), f"legitimate accounts were flagged: {sorted(flagged)}"

    def test_a_clean_ledger_flags_nothing(self):
        scores = _zscore_anomaly_scores(_baseline_transactions())

        assert scores.max() <= 2.0, "an unmodified ledger must not trip the detector"

    def test_scoring_is_stable_across_runs(self):
        base = _baseline_transactions()
        augmented, _ = inject_synthetic_fraud(base, seed=SEED)

        first = _zscore_anomaly_scores(augmented)
        second = _zscore_anomaly_scores(augmented)

        np.testing.assert_allclose(first, second)

    def test_scores_are_centred_and_scaled(self):
        scores = _zscore_anomaly_scores(_baseline_transactions())

        assert abs(float(scores.mean())) < 1e-9
        assert abs(float(scores.std()) - 1.0) < 1e-9 or float(scores.std()) == 0.0


def _miscalibrated_scores(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Over-confident scores whose true event rate is lower than claimed."""
    rng = np.random.default_rng(SEED)
    y_prob = rng.uniform(0.0, 1.0, size=n)
    # True probability is squashed relative to the reported score, so the raw
    # scores systematically overstate risk — the classic calibration target.
    y_true = (rng.uniform(0.0, 1.0, size=n) < y_prob**2).astype(int)
    return y_prob, y_true


class TestPlattCalibration:
    def test_calibration_reduces_brier_score(self):
        y_prob, y_true = _miscalibrated_scores()
        calibrator = PlattCalibrator().fit(y_prob, y_true)

        raw = float(np.mean((y_prob - y_true) ** 2))
        calibrated = float(np.mean((calibrator.calibrate(y_prob) - y_true) ** 2))

        assert calibrated < raw, "calibration must improve the Brier score"

    def test_output_stays_within_probability_bounds(self):
        y_prob, y_true = _miscalibrated_scores()
        calibrated = PlattCalibrator().fit(y_prob, y_true).calibrate(y_prob)

        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_calibration_is_monotonic_in_the_raw_score(self):
        y_prob, y_true = _miscalibrated_scores()
        calibrator = PlattCalibrator().fit(y_prob, y_true)

        grid = np.linspace(0.01, 0.99, 50)
        out = calibrator.calibrate(grid)

        # Platt scaling is a monotone transform; losing that would silently
        # reorder ranked predictions.
        assert np.all(np.diff(out) >= -1e-9)

    def test_refitting_on_the_same_data_is_stable(self):
        y_prob, y_true = _miscalibrated_scores()

        first = PlattCalibrator().fit(y_prob, y_true)
        second = PlattCalibrator().fit(y_prob, y_true)

        assert first.a == pytest.approx(second.a)
        assert first.b == pytest.approx(second.b)

    def test_calibrating_before_fitting_raises(self):
        with pytest.raises(RuntimeError):
            PlattCalibrator().calibrate(np.array([0.5]))

    def test_mismatched_input_lengths_raise(self):
        with pytest.raises(ValueError):
            PlattCalibrator().fit(np.array([0.1, 0.2]), np.array([1]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            PlattCalibrator().fit(np.array([]), np.array([]))


class TestIsotonicCalibration:
    def test_calibration_reduces_brier_score(self):
        y_prob, y_true = _miscalibrated_scores()
        calibrator = IsotonicCalibrator().fit(y_prob, y_true)

        raw = float(np.mean((y_prob - y_true) ** 2))
        calibrated = float(np.mean((calibrator.calibrate(y_prob) - y_true) ** 2))

        assert calibrated < raw

    def test_output_respects_the_configured_bounds(self):
        y_prob, y_true = _miscalibrated_scores()
        calibrated = IsotonicCalibrator(y_min=0.0, y_max=1.0).fit(y_prob, y_true).calibrate(y_prob)

        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_calibration_is_non_decreasing(self):
        y_prob, y_true = _miscalibrated_scores()
        calibrator = IsotonicCalibrator().fit(y_prob, y_true)

        grid = np.linspace(0.01, 0.99, 50)
        out = calibrator.calibrate(grid)

        assert np.all(np.diff(out) >= -1e-9)

    def test_is_deterministic(self):
        y_prob, y_true = _miscalibrated_scores()

        first = IsotonicCalibrator().fit(y_prob, y_true).calibrate(y_prob)
        second = IsotonicCalibrator().fit(y_prob, y_true).calibrate(y_prob)

        np.testing.assert_allclose(first, second)

    def test_calibrating_before_fitting_raises(self):
        with pytest.raises(RuntimeError):
            IsotonicCalibrator().calibrate(np.array([0.5]))


class TestCalibrationOnDegenerateInput:
    def test_a_single_class_does_not_crash_platt(self):
        y_prob = np.linspace(0.1, 0.9, 20)
        y_true = np.ones(20, dtype=int)

        try:
            calibrated = PlattCalibrator().fit(y_prob, y_true).calibrate(y_prob)
        except (ValueError, RuntimeError):
            # An explicit refusal is an acceptable contract; a crash deep in the
            # numerics is not.
            return

        assert np.all(np.isfinite(calibrated))

    def test_constant_scores_do_not_produce_nan(self):
        y_prob = np.full(20, 0.5)
        y_true = np.array([0, 1] * 10)

        calibrated = IsotonicCalibrator().fit(y_prob, y_true).calibrate(y_prob)

        assert np.all(np.isfinite(calibrated))
