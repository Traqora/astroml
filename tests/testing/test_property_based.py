"""Property-based tests for ML model testing infrastructure.

Issue #658: Validates the four modules in
``astroml/testing/property_based/`` against realistic ML data produced by
Hypothesis strategies.

Coverage targets (>90% for changed code):
- strategies.py
- model_properties.py
- data_properties.py
- invariant_checker.py
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from astroml.testing.property_based.data_properties import (
    DataPropertyValidator,
    check_class_labels_in_range,
    check_feature_matrix_shape,
    check_feature_variance,
    check_label_array_shape,
    check_minimum_samples,
    check_no_nan_or_inf,
)
from astroml.testing.property_based.invariant_checker import (
    InvariantChecker,
    InvariantViolation,
    create_standard_invariant_checker,
)
from astroml.testing.property_based.model_properties import (
    BUILTIN_PROPERTIES,
    ModelPropertyRunner,
    PropertyCheckResult,
    binary_predictions_in_set,
    output_shape_matches_samples,
    outputs_are_finite,
    probability_rows_sum_to_one,
    scores_in_unit_interval,
)
from astroml.testing.property_based.strategies import (
    binary_label_strategy,
    feature_matrix_strategy,
    multiclass_label_strategy,
    probability_output_strategy,
    scalar_score_strategy,
)

# ── Shared Hypothesis settings ────────────────────────────────────────────────

_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    phases=[Phase.generate, Phase.shrink],
    suppress_health_check=[HealthCheck.too_slow],
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. strategies.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureMatrixStrategy:
    """Tests for :func:`feature_matrix_strategy`."""

    @given(
        X=feature_matrix_strategy(min_samples=1, max_samples=20, min_features=2, max_features=10)
    )
    @_SETTINGS
    def test_shape_within_bounds(self, X: np.ndarray) -> None:
        assert X.ndim == 2
        assert 1 <= X.shape[0] <= 20
        assert 2 <= X.shape[1] <= 10

    @given(X=feature_matrix_strategy())
    @_SETTINGS
    def test_dtype_is_float32(self, X: np.ndarray) -> None:
        assert X.dtype == np.float32

    @given(X=feature_matrix_strategy())
    @_SETTINGS
    def test_all_finite(self, X: np.ndarray) -> None:
        assert np.all(np.isfinite(X))


class TestLabelStrategies:
    """Tests for binary / multiclass label strategies."""

    @given(y=binary_label_strategy(n_samples=10))
    @_SETTINGS
    def test_binary_labels_shape(self, y: np.ndarray) -> None:
        assert y.shape == (10,)
        assert np.all(np.isin(y, [0, 1]))

    @given(y=multiclass_label_strategy(n_samples=15, n_classes=4))
    @_SETTINGS
    def test_multiclass_labels_in_range(self, y: np.ndarray) -> None:
        assert y.shape == (15,)
        assert np.all((y >= 0) & (y < 4))


class TestProbabilityOutputStrategy:
    """Tests for :func:`probability_output_strategy`."""

    @given(probs=probability_output_strategy(n_samples=5, n_classes=3))
    @_SETTINGS
    def test_rows_sum_to_one(self, probs: np.ndarray) -> None:
        assert probs.shape == (5, 3)
        assert np.all(np.abs(probs.sum(axis=1) - 1.0) < 1e-9)

    @given(probs=probability_output_strategy(n_samples=8, n_classes=2))
    @_SETTINGS
    def test_values_non_negative(self, probs: np.ndarray) -> None:
        assert np.all(probs >= 0.0)


class TestScalarScoreStrategy:
    """Tests for :func:`scalar_score_strategy`."""

    @given(score=scalar_score_strategy())
    @_SETTINGS
    def test_score_in_unit_interval(self, score: float) -> None:
        assert 0.0 <= score <= 1.0

    @given(score=scalar_score_strategy(min_value=-5.0, max_value=5.0))
    @_SETTINGS
    def test_custom_bounds(self, score: float) -> None:
        assert -5.0 <= score <= 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. model_properties.py — built-in invariants
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuiltinProperties:
    """Unit tests for the built-in property functions."""

    def test_outputs_are_finite_passes(self) -> None:
        X = np.zeros((3, 2), dtype=np.float32)
        output = np.array([0.1, 0.5, 0.9])
        assert outputs_are_finite(X, output) is True

    def test_outputs_are_finite_fails_on_nan(self) -> None:
        X = np.zeros((2, 2), dtype=np.float32)
        output = np.array([0.5, float("nan")])
        assert outputs_are_finite(X, output) is False

    def test_outputs_are_finite_fails_on_inf(self) -> None:
        X = np.zeros((2, 2), dtype=np.float32)
        output = np.array([float("inf"), 0.3])
        assert outputs_are_finite(X, output) is False

    def test_output_shape_matches_samples_pass(self) -> None:
        X = np.zeros((5, 3), dtype=np.float32)
        output = np.zeros((5, 2))
        assert output_shape_matches_samples(X, output) is True

    def test_output_shape_matches_samples_fail(self) -> None:
        X = np.zeros((5, 3), dtype=np.float32)
        output = np.zeros((4, 2))
        assert output_shape_matches_samples(X, output) is False

    def test_binary_predictions_in_set_pass(self) -> None:
        X = np.zeros((4, 2), dtype=np.float32)
        output = np.array([0, 1, 0, 1])
        assert binary_predictions_in_set(X, output) is True

    def test_binary_predictions_in_set_fail(self) -> None:
        X = np.zeros((3, 2), dtype=np.float32)
        output = np.array([0, 1, 2])
        assert binary_predictions_in_set(X, output) is False

    def test_probability_rows_sum_to_one_pass(self) -> None:
        X = np.zeros((2, 2), dtype=np.float32)
        output = np.array([[0.3, 0.7], [0.5, 0.5]])
        assert probability_rows_sum_to_one(X, output) is True

    def test_probability_rows_sum_to_one_fails_1d(self) -> None:
        X = np.zeros((2, 2), dtype=np.float32)
        output = np.array([0.5, 0.5])
        assert probability_rows_sum_to_one(X, output) is False

    def test_probability_rows_sum_to_one_fails_wrong_sum(self) -> None:
        X = np.zeros((1, 2), dtype=np.float32)
        output = np.array([[0.4, 0.4]])  # sums to 0.8
        assert probability_rows_sum_to_one(X, output) is False

    def test_scores_in_unit_interval_pass(self) -> None:
        X = np.zeros((3, 2), dtype=np.float32)
        output = np.array([0.0, 0.5, 1.0])
        assert scores_in_unit_interval(X, output) is True

    def test_scores_in_unit_interval_fail(self) -> None:
        X = np.zeros((2, 2), dtype=np.float32)
        output = np.array([0.5, 1.5])
        assert scores_in_unit_interval(X, output) is False

    def test_builtin_registry_completeness(self) -> None:
        expected = {
            "outputs_are_finite",
            "output_shape_matches_samples",
            "binary_predictions_in_set",
            "probability_rows_sum_to_one",
            "scores_in_unit_interval",
        }
        assert set(BUILTIN_PROPERTIES.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 3. model_properties.py — ModelPropertyRunner
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelPropertyRunner:
    """Integration tests for :class:`ModelPropertyRunner`."""

    def _identity_model(self, X: np.ndarray) -> np.ndarray:
        """Model that returns a column of zeros—always finite, shape-correct."""
        return np.zeros(X.shape[0], dtype=np.float32)

    def _nan_model(self, X: np.ndarray) -> np.ndarray:
        """Model that always injects NaN."""
        out = np.zeros(X.shape[0], dtype=np.float32)
        out[0] = float("nan")
        return out

    def test_runner_passes_for_valid_model(self) -> None:
        runner = ModelPropertyRunner(
            model=self._identity_model,
            properties=[outputs_are_finite, output_shape_matches_samples],
            n_features=4,
        )
        results = runner.run(max_examples=20)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_runner_detects_nan_output(self) -> None:
        runner = ModelPropertyRunner(
            model=self._nan_model,
            properties=[outputs_are_finite],
            n_features=4,
        )
        results = runner.run(max_examples=10)
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].property_name == "outputs_are_finite"

    def test_runner_results_property(self) -> None:
        runner = ModelPropertyRunner(
            model=self._identity_model,
            properties=[outputs_are_finite],
            n_features=3,
        )
        runner.run(max_examples=5)
        assert len(runner.results) == 1

    def test_property_check_result_fields(self) -> None:
        result = PropertyCheckResult(property_name="test_prop", passed=True)
        assert result.property_name == "test_prop"
        assert result.passed is True
        assert result.violation_input is None
        assert result.error_message is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. data_properties.py — individual checks
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataPropertyChecks:
    """Unit tests for individual data validation functions."""

    # check_no_nan_or_inf
    def test_no_nan_or_inf_pass(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = check_no_nan_or_inf(X)
        assert result.passed is True

    def test_no_nan_or_inf_fail_nan(self) -> None:
        X = np.array([[1.0, float("nan")]])
        result = check_no_nan_or_inf(X)
        assert result.passed is False
        assert "NaN" in result.error_message  # type: ignore[operator]

    def test_no_nan_or_inf_fail_inf(self) -> None:
        X = np.array([[float("inf"), 1.0]])
        result = check_no_nan_or_inf(X)
        assert result.passed is False

    # check_feature_matrix_shape
    def test_feature_matrix_shape_pass(self) -> None:
        X = np.zeros((10, 5))
        result = check_feature_matrix_shape(X, expected_features=5)
        assert result.passed is True

    def test_feature_matrix_shape_fail_wrong_dim(self) -> None:
        X = np.zeros((10,))
        result = check_feature_matrix_shape(X)
        assert result.passed is False

    def test_feature_matrix_shape_fail_wrong_features(self) -> None:
        X = np.zeros((10, 4))
        result = check_feature_matrix_shape(X, expected_features=5)
        assert result.passed is False

    # check_label_array_shape
    def test_label_array_shape_pass(self) -> None:
        X = np.zeros((5, 3))
        y = np.array([0, 1, 0, 1, 0])
        result = check_label_array_shape(y, X)
        assert result.passed is True

    def test_label_array_shape_fail_mismatch(self) -> None:
        X = np.zeros((5, 3))
        y = np.array([0, 1])
        result = check_label_array_shape(y, X)
        assert result.passed is False

    def test_label_array_shape_fail_2d(self) -> None:
        X = np.zeros((3, 3))
        y = np.zeros((3, 2))
        result = check_label_array_shape(y, X)
        assert result.passed is False

    # check_class_labels_in_range
    def test_class_labels_in_range_pass(self) -> None:
        y = np.array([0, 1, 2, 0])
        result = check_class_labels_in_range(y, n_classes=3)
        assert result.passed is True

    def test_class_labels_in_range_fail_negative(self) -> None:
        y = np.array([-1, 0, 1])
        result = check_class_labels_in_range(y, n_classes=2)
        assert result.passed is False

    def test_class_labels_in_range_fail_too_large(self) -> None:
        y = np.array([0, 1, 5])
        result = check_class_labels_in_range(y, n_classes=3)
        assert result.passed is False

    # check_minimum_samples
    def test_minimum_samples_pass(self) -> None:
        X = np.zeros((10, 3))
        result = check_minimum_samples(X, min_samples=5)
        assert result.passed is True

    def test_minimum_samples_fail(self) -> None:
        X = np.zeros((2, 3))
        result = check_minimum_samples(X, min_samples=5)
        assert result.passed is False

    # check_feature_variance
    def test_feature_variance_pass(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = check_feature_variance(X, min_variance=0.0)
        assert result.passed is True

    def test_feature_variance_fail_constant_column(self) -> None:
        X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        result = check_feature_variance(X, min_variance=0.0)
        assert result.passed is False
        assert "0" in result.error_message  # type: ignore[operator]

    def test_feature_variance_single_sample_skipped(self) -> None:
        X = np.array([[1.0, 1.0]])
        result = check_feature_variance(X)
        assert result.passed is True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. data_properties.py — DataPropertyValidator
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataPropertyValidator:
    """Tests for :class:`DataPropertyValidator`."""

    def test_all_passed_on_clean_data(self) -> None:
        validator = DataPropertyValidator(expected_features=3, n_classes=2)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.array([0, 1] * 10, dtype=np.int64)
        assert validator.all_passed(X, y) is True

    def test_fails_on_nan(self) -> None:
        validator = DataPropertyValidator()
        X = np.array([[1.0, float("nan")], [2.0, 3.0]])
        results = validator.validate(X)
        failed = [r for r in results if not r.passed]
        assert any(r.check_name == "no_nan_or_inf" for r in failed)

    def test_validate_without_labels(self) -> None:
        validator = DataPropertyValidator(expected_features=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        results = validator.validate(X)
        assert all(r.passed for r in results)

    def test_validate_wrong_feature_count(self) -> None:
        validator = DataPropertyValidator(expected_features=5)
        X = np.zeros((10, 3))
        results = validator.validate(X)
        failed_names = [r.check_name for r in results if not r.passed]
        assert "feature_matrix_shape" in failed_names

    def test_all_passed_returns_false_on_bad_data(self) -> None:
        validator = DataPropertyValidator(min_samples=100)
        X = np.zeros((5, 3))
        assert validator.all_passed(X) is False

    @given(X=feature_matrix_strategy(min_samples=2, max_samples=30, min_features=3, max_features=3))
    @_SETTINGS
    def test_hypothesis_clean_matrix_passes(self, X: np.ndarray) -> None:
        validator = DataPropertyValidator(expected_features=3)
        results = validator.validate(X)
        # feature_variance may legitimately fail for constant matrices generated
        # by the strategy (all-zeros); that check is unit-tested separately.
        for r in results:
            if r.check_name == "feature_variance":
                continue
            assert r.passed, f"Check '{r.check_name}' failed: {r.error_message}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. invariant_checker.py — InvariantChecker
# ═══════════════════════════════════════════════════════════════════════════════


class TestInvariantChecker:
    """Tests for :class:`InvariantChecker`."""

    def _make_checker(self) -> InvariantChecker:
        checker = InvariantChecker()
        checker.register("finite", lambda X, y: bool(np.all(np.isfinite(y))))
        return checker

    def test_check_passes_for_finite_output(self) -> None:
        checker = self._make_checker()
        X = np.zeros((3, 2))
        output = np.array([0.1, 0.5, 0.9])
        violations = checker.check(X, output)
        assert violations == []

    def test_check_detects_nan_output(self) -> None:
        checker = self._make_checker()
        X = np.zeros((2, 2))
        output = np.array([0.5, float("nan")])
        violations = checker.check(X, output)
        assert len(violations) == 1
        assert violations[0].invariant_name == "finite"

    def test_register_duplicate_raises(self) -> None:
        checker = self._make_checker()
        with pytest.raises(ValueError, match="already registered"):
            checker.register("finite", lambda X, y: True)

    def test_remove_invariant(self) -> None:
        checker = self._make_checker()
        checker.remove("finite")
        assert "finite" not in checker.registered_names

    def test_remove_missing_raises(self) -> None:
        checker = InvariantChecker()
        with pytest.raises(KeyError):
            checker.remove("nonexistent")

    def test_registered_names_property(self) -> None:
        checker = self._make_checker()
        assert "finite" in checker.registered_names

    def test_violations_property_after_check(self) -> None:
        checker = self._make_checker()
        X = np.zeros((1, 2))
        output = np.array([float("nan")])
        checker.check(X, output)
        assert len(checker.violations) == 1

    def test_check_per_sample_sets_index(self) -> None:
        checker = InvariantChecker()
        checker.register("non_negative", lambda X, y: bool(np.all(y >= 0)))
        X = np.zeros((3, 2))
        output = np.array([1.0, -1.0, 2.0])
        violations = checker.check_per_sample(X, output)
        assert len(violations) == 1
        assert violations[0].sample_index == 1

    def test_invariant_violation_fields(self) -> None:
        X = np.zeros((1, 2))
        output = np.array([0.5])
        v = InvariantViolation(
            invariant_name="my_inv",
            sample_index=0,
            input_sample=X,
            model_output=output,
            description="test violation",
        )
        assert v.invariant_name == "my_inv"
        assert v.sample_index == 0
        assert v.description == "test violation"

    def test_exception_in_invariant_is_captured(self) -> None:
        checker = InvariantChecker()

        def boom(X: np.ndarray, y: np.ndarray) -> bool:
            raise RuntimeError("deliberate failure")

        checker.register("boom", boom)
        X = np.zeros((2, 2))
        output = np.zeros(2)
        violations = checker.check(X, output)
        assert len(violations) == 1
        assert "deliberate failure" in violations[0].description


class TestStandardInvariantChecker:
    """Tests for :func:`create_standard_invariant_checker`."""

    def test_factory_registers_expected_invariants(self) -> None:
        checker = create_standard_invariant_checker()
        assert "finite_outputs" in checker.registered_names
        assert "non_negative_probabilities" in checker.registered_names
        assert "output_shape_consistent" in checker.registered_names

    def test_standard_checker_passes_valid_output(self) -> None:
        checker = create_standard_invariant_checker()
        X = np.zeros((5, 4))
        output = np.full((5, 2), 0.5)
        violations = checker.check(X, output)
        assert violations == []

    def test_standard_checker_fails_inf_output(self) -> None:
        checker = create_standard_invariant_checker()
        X = np.zeros((2, 4))
        output = np.array([[float("inf"), 0.0], [0.5, 0.5]])
        violations = checker.check(X, output)
        assert any(v.invariant_name == "finite_outputs" for v in violations)

    def test_standard_checker_fails_wrong_shape(self) -> None:
        checker = create_standard_invariant_checker()
        X = np.zeros((3, 4))
        output = np.zeros((2, 2))
        violations = checker.check(X, output)
        assert any(v.invariant_name == "output_shape_consistent" for v in violations)

    @given(X=feature_matrix_strategy(min_samples=1, max_samples=20, min_features=4, max_features=4))
    @_SETTINGS
    def test_hypothesis_zero_output_passes(self, X: np.ndarray) -> None:
        checker = create_standard_invariant_checker()
        output = np.zeros((X.shape[0], 2), dtype=np.float64)
        violations = checker.check(X, output)
        assert violations == [], f"Unexpected violations: {violations}"
