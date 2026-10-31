"""Tests for the LLM eval CI quality-gate threshold logic (issue #875).

``scripts/ci/run-eval.sh`` used to compute a ``SCORE_THRESHOLD`` value and
only ever print it — never actually compare any real score against it —
while also piping the real eval test run through ``|| true``, so the CI
"quality gate" step for LLM evaluation always exited 0 regardless of the
actual eval outcome. ``astroml/llm/eval/threshold.py`` extracts the
score-vs-threshold comparison into a small, directly testable Python
function/dataclass instead of leaving it as inline shell, which is what
this module exercises.
"""

from __future__ import annotations

import pytest

from astroml.llm.eval.threshold import (
    ThresholdCheckResult,
    check_threshold,
    check_threshold_from_report_dict,
    compute_overall_score,
)


def test_compute_overall_score_perfect_metrics() -> None:
    assert compute_overall_score({"bleu": 1.0, "rouge_l": 1.0, "safety": 1.0}) == 10.0


def test_compute_overall_score_zero_metrics_value() -> None:
    assert compute_overall_score({"bleu": 0.0, "rouge_l": 0.0}) == 0.0


def test_compute_overall_score_mixed_metrics() -> None:
    # mean(0.8, 0.6, 1.0) = 0.8 -> 8.0 on the 0-10 scale
    score = compute_overall_score({"bleu": 0.8, "rouge_l": 0.6, "safety": 1.0})
    assert score == pytest.approx(8.0)


def test_compute_overall_score_empty_metrics_is_zero() -> None:
    """An eval run that produced no scorable metrics (e.g. every case
    errored) must not be treated as vacuously passing."""
    assert compute_overall_score({}) == 0.0


@pytest.mark.parametrize(
    ("metrics", "threshold", "expected_passed"),
    [
        ({"bleu": 0.9, "safety": 1.0}, 5.0, True),
        ({"bleu": 0.1, "safety": 0.2}, 5.0, False),
        # Exactly at the threshold boundary should pass (>=, not strictly >).
        ({"bleu": 0.5, "safety": 0.5}, 5.0, True),
        ({}, 5.0, False),
        ({"safety": 1.0}, 0.0, True),
    ],
)
def test_check_threshold_pass_fail(
    metrics: dict[str, float], threshold: float, expected_passed: bool
) -> None:
    result = check_threshold(metrics, threshold)
    assert isinstance(result, ThresholdCheckResult)
    assert result.passed is expected_passed
    assert result.threshold == threshold
    assert result.metric_count == len(metrics)


def test_check_threshold_result_summary_reports_pass_or_fail() -> None:
    passing = check_threshold({"bleu": 1.0}, threshold=5.0)
    failing = check_threshold({"bleu": 0.0}, threshold=5.0)

    assert "[PASS]" in passing.summary()
    assert "[FAIL]" in failing.summary()
    assert "threshold" in passing.summary().lower()


def test_check_threshold_from_report_dict_uses_metrics_key() -> None:
    report = {
        "model_name": "ci-eval-run",
        "metrics": {"bleu": 0.95, "rouge_l": 0.9, "safety": 1.0},
        "has_regression": False,
    }
    result = check_threshold_from_report_dict(report, threshold=5.0)
    assert result.passed is True
    assert result.score == pytest.approx(9.5)


def test_check_threshold_from_report_dict_missing_metrics_key_fails_closed() -> None:
    """A report with no "metrics" key at all must be treated as zero
    metrics (score 0.0, fails the gate) rather than raising or silently
    passing."""
    result = check_threshold_from_report_dict({}, threshold=5.0)
    assert result.passed is False
    assert result.score == 0.0


def test_check_threshold_from_report_dict_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        check_threshold_from_report_dict({"metrics": ["not", "a", "dict"]}, threshold=5.0)


def test_threshold_boundary_is_inclusive_not_exclusive() -> None:
    """A score exactly equal to the threshold must PASS the gate, not fail
    it — this is the kind of off-by-one a careless rewrite of the shell
    comparison could easily introduce."""
    exact = check_threshold({"bleu": 0.5}, threshold=5.0)
    assert exact.passed is True

    just_below = check_threshold({"bleu": 0.4999}, threshold=5.0)
    assert just_below.passed is False
