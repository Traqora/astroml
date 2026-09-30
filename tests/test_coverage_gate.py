"""Gate-configuration tests for issue #716.

These tests do not measure coverage themselves — they lock in the *gate*:
the pyproject floor, the CI flag that enforces it, and the alignment with
the codecov patch target. If someone weakens or drops the gate, this file
fails first with a message that names the exact file to fix.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PYTEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pytest.yml"
CODECOV_YML = REPO_ROOT / "codecov.yml"

# Single source of truth for the floor asserted below. Mirrors
# [tool.coverage.report] fail_under in pyproject.toml and the codecov
# patch target in codecov.yml.
EXPECTED_FAIL_UNDER = 70


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


class TestCoverageGateConfig:
    def test_pyproject_declares_fail_under(self) -> None:
        report = _pyproject()["tool"]["coverage"]["report"]
        assert report.get("fail_under", 0) >= EXPECTED_FAIL_UNDER, (
            "pyproject [tool.coverage.report] fail_under was lowered or removed; "
            f"expected >= {EXPECTED_FAIL_UNDER}"
        )

    def test_pyproject_shows_missing_lines(self) -> None:
        report = _pyproject()["tool"]["coverage"]["report"]
        assert report.get("show_missing") is True, (
            "show_missing must stay on so breach reports name the uncovered lines"
        )

    def test_ci_enforces_cov_fail_under(self) -> None:
        text = PYTEST_WORKFLOW.read_text()
        assert "--cov-fail-under=70" in text, (
            ".github/workflows/pytest.yml must pass --cov-fail-under so the "
            "gate is enforced (and visible) in CI logs"
        )

    def test_ci_prints_missing_lines_report(self) -> None:
        text = PYTEST_WORKFLOW.read_text()
        assert "coverage report --show-missing" in text, (
            "CI must print the per-file missing-lines table on every run so a "
            "breach tells the author exactly what to cover"
        )

    def test_codecov_patch_target_alignment(self) -> None:
        text = CODECOV_YML.read_text()
        assert "target: 70%" in text, (
            "codecov patch target drifted from the CI floor; keep them aligned"
        )
