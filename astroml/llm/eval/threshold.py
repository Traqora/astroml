"""Overall LLM eval score computation and CI quality-gate enforcement (#875).

``scripts/ci/run-eval.sh`` previously computed a ``SCORE_THRESHOLD`` value
and then only ever *printed* it — it never compared any actual score
against it, and separately piped the real eval test run through
``|| true``, so the CI "quality gate" step always exited 0 regardless of
whether the LLM evaluation suite passed, failed, or scored anything at
all. This module provides the single piece that was missing: a real,
testable score-vs-threshold comparison, built on top of the metrics this
codebase's :class:`astroml.llm.eval.framework.LLMEvalFramework` already
computes (bleu, rouge_l, factuality, relevance, coherence, safety — each on
a 0.0-1.0 scale per :mod:`astroml.llm.eval.metrics`), rather than inventing
a new scoring mechanism.

Scale: the eval metrics in this codebase are 0.0-1.0. The CI config's
``SCORE_THRESHOLD`` default of ``5.0`` only makes sense against a 0-10
scale, so :func:`compute_overall_score` reports the mean of the averaged
per-metric scores scaled to 0-10 (i.e. a perfect run scores 10.0, matching
the "score out of 10" convention ``SCORE_THRESHOLD=5.0`` implies).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Metrics scale 0.0-1.0; the CI threshold is expressed on a 0-10 scale.
_SCORE_SCALE = 10.0


@dataclass(frozen=True)
class ThresholdCheckResult:
    """Outcome of comparing a computed eval score against its threshold."""

    score: float
    threshold: float
    passed: bool
    metric_count: int

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] overall score {self.score:.4f} "
            f"{'>=' if self.passed else '<'} threshold {self.threshold:.4f} "
            f"(from {self.metric_count} metric(s))"
        )


def compute_overall_score(metrics: dict[str, float]) -> float:
    """Reduce a per-metric averages dict (as returned in
    ``LLMEvalFramework.run_full_evaluation()["metrics"]``) to a single
    overall score on a 0-10 scale.

    Args:
        metrics: Mapping of metric name -> averaged score in [0.0, 1.0],
            e.g. ``{"bleu": 0.81, "rouge_l": 0.77, "safety": 1.0}``.

    Returns:
        The mean of ``metrics.values()`` scaled to 0-10. ``0.0`` if
        ``metrics`` is empty (e.g. every benchmark case errored out), which
        is intentionally a failing score rather than treated as "no data,
        skip the gate" — an eval run that produced nothing to score is not
        safe to treat as passing.
    """
    if not metrics:
        return 0.0
    return (sum(metrics.values()) / len(metrics)) * _SCORE_SCALE


def check_threshold(metrics: dict[str, float], threshold: float) -> ThresholdCheckResult:
    """Compute the overall score from *metrics* and compare it to *threshold*.

    Args:
        metrics: Per-metric averages, as produced by the eval framework.
        threshold: Minimum acceptable overall score (0-10 scale).

    Returns:
        A :class:`ThresholdCheckResult` describing the outcome.
    """
    score = compute_overall_score(metrics)
    return ThresholdCheckResult(
        score=score,
        threshold=threshold,
        passed=score >= threshold,
        metric_count=len(metrics),
    )


def check_threshold_from_report_dict(
    report: dict[str, object], threshold: float
) -> ThresholdCheckResult:
    """Convenience wrapper around :func:`check_threshold` for callers that
    hold the full ``run_full_evaluation()`` result dict rather than just its
    ``"metrics"`` entry (e.g. the CI script, once it's run through Python).
    """
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError(f"Expected report['metrics'] to be a dict, got {type(metrics)!r}")
    return check_threshold(metrics, threshold)
