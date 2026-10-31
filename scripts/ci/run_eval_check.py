#!/usr/bin/env python3
"""CI driver: run the LLM eval framework and enforce the quality gate (#875).

This is the piece ``scripts/ci/run-eval.sh`` was missing: it actually runs
an evaluation through :class:`astroml.llm.eval.framework.LLMEvalFramework`
(the existing scoring mechanism — see ``astroml/llm/eval/metrics.py`` for
the underlying BLEU/ROUGE-L/factuality/relevance/coherence/safety
calculations), reduces the result to a single overall score via
:func:`astroml.llm.eval.threshold.check_threshold`, and exits non-zero when
that score is below ``SCORE_THRESHOLD`` — which the shell script previously
computed but never actually compared against anything.

In the absence of a configured real LLM provider in CI, this uses the
existing ``DeterministicMockProvider`` test double (the same mechanism
``tests/llm/eval/test_eval.py::test_evaluation_framework`` already uses) so
the gate is still exercising the real eval framework and metrics end to
end. Once a real provider is wired up for CI, swap ``_build_generation_fn``
to call it instead — the threshold-enforcement logic itself does not change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable

from astroml.llm.eval.framework import LLMEvalFramework
from astroml.llm.eval.threshold import check_threshold_from_report_dict


def _build_generation_fn() -> Callable[[str], Awaitable[str]]:
    """Return an async ``generation_fn`` for :class:`LLMEvalFramework`.

    Uses the repo's existing deterministic mock provider (see module
    docstring) rather than inventing a new one.
    """
    from tests.llm.mocks import DeterministicMockProvider

    provider = DeterministicMockProvider(latency_ms=0.0)

    async def generation_fn(prompt: str) -> str:
        return str(provider.generate(prompt))

    return generation_fn


async def _run(threshold: float, save_baseline: bool) -> int:
    framework = LLMEvalFramework("ci-eval-run", _build_generation_fn())
    report = await framework.run_full_evaluation(save_baseline=save_baseline)

    result = check_threshold_from_report_dict(report, threshold)

    print(
        json.dumps({"metrics": report["metrics"], "report_path": report["report_path"]}, indent=2)
    )
    print(result.summary())

    if report["has_regression"]:
        print("Quality regression detected vs. saved baseline:")
        for detail in report["regressions"]:
            print(f"  - {detail}")

    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM eval and enforce the score threshold.")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Minimum acceptable overall score on a 0-10 scale.",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save this run's metrics as the new regression baseline if it passes.",
    )
    args = parser.parse_args(argv)

    return asyncio.run(_run(args.threshold, args.save_baseline))


if __name__ == "__main__":
    raise SystemExit(main())
