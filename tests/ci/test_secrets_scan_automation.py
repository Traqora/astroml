"""Automation wiring tests for secrets scanning — issue #938.

A secret scan only prevents leaks if it actually runs and can fail the build.
These tests pin the CI/pre-commit wiring: both scanners are registered, the
mutating raw scan is not used as a gate, the gate runs on every push/PR, the
full-history scan is scheduled, and every baseline entry is audited as a
false positive.
"""

from __future__ import annotations

import json
from pathlib import Path

from astroml.ci.secrets_gate import evaluate

ROOT = Path(__file__).resolve().parents[2]
PRE_COMMIT_CONFIG = ROOT / ".pre-commit-config.yaml"
PRE_COMMIT_WORKFLOW = ROOT / ".github" / "workflows" / "pre-commit.yml"
SECURITY_WORKFLOW = ROOT / ".github" / "workflows" / "security.yml"
MAKEFILE = ROOT / "Makefile"
BASELINE = ROOT / ".secrets.baseline"


def _text(path: Path) -> str:
    assert path.exists(), f"{path.relative_to(ROOT)} is missing"
    return path.read_text(encoding="utf-8")


def test_pre_commit_runs_detect_secrets_and_the_gate() -> None:
    config = _text(PRE_COMMIT_CONFIG)

    assert "id: detect-secrets" in config
    assert "id: secrets-gate" in config
    assert "astroml.ci.secrets_gate" in config


def test_ci_workflow_invokes_the_failing_gate() -> None:
    workflow = _text(PRE_COMMIT_WORKFLOW)

    assert "python -m astroml.ci.secrets_gate" in workflow
    assert "--baseline .secrets.baseline" in workflow


def test_ci_workflow_does_not_use_the_mutating_raw_scan() -> None:
    workflow = _text(PRE_COMMIT_WORKFLOW)

    assert "detect-secrets scan --baseline" not in workflow, (
        "raw `detect-secrets scan --baseline` rewrites the baseline and exits 0; "
        "use astroml.ci.secrets_gate instead"
    )


def test_full_history_scan_is_scheduled() -> None:
    workflow = _text(SECURITY_WORKFLOW)

    assert "schedule:" in workflow
    assert "gitleaks-action" in workflow


def test_make_target_runs_the_gate() -> None:
    makefile = _text(MAKEFILE)
    target = makefile.split("secrets-scan:", 1)[1]

    assert "astroml.ci.secrets_gate" in target


def test_baseline_is_audited_and_free_of_real_secrets() -> None:
    baseline = json.loads(_text(BASELINE))

    report = evaluate(baseline, baseline)

    assert report.ok, report.format()
    assert report.new_secrets == []
    assert report.unaudited_secrets == []
    assert report.true_positive_secrets == []
