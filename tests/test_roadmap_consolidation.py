"""Tests for scripts/check_roadmap_consolidation.py.

"Unsupported feature claims" used to live inline as ``## Future Enhancements``
sections in feature docs and as stale to-dos in the README roadmap. These tests
pin the consolidation into docs/ROADMAP.md so the v1 onboarding path only ever
describes what works on a clean machine.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "check_roadmap_consolidation.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_roadmap_consolidation", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


@pytest.fixture
def repo_with_roadmap(tmp_path: Path) -> Path:
    """A minimal repo layout that satisfies the consolidation rules."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "DATA_QUALITY_VALIDATION.md").write_text(
        "# Data Quality\n\nSee [ROADMAP.md](ROADMAP.md).\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "FRAUD_REGISTRY_CONTRACT.md").write_text(
        "# Fraud Registry\n\nSee [ROADMAP.md](ROADMAP.md).\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "ROADMAP.md").write_text(
        "# Roadmap\n\n"
        + "\n".join(f"- {claim}" for claim in checker.ROADMAP_CLAIMS)
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# AstroML\n\n## Roadmap\n\n- [ ] Real-time streaming\n", encoding="utf-8"
    )
    return tmp_path


class TestFutureClaimsMoved:
    def test_feature_docs_have_no_inline_future_claims(self) -> None:
        # Happy path: the two feature docs no longer carry inline
        # Future Enhancements / Planned Features headings.
        for rel in checker.CONSOLIDATED_DOCS:
            text = (ROOT / rel).read_text(encoding="utf-8")
            assert checker.find_unconsolidated_future_claims(text) == [], rel

    def test_inline_future_claim_is_detected(self) -> None:
        # Failure path: a doc that still carries the heading is reported.
        text = "# X\n\n## Future Enhancements\n\n- something planned\n"
        assert checker.find_unconsolidated_future_claims(text) == ["## Future Enhancements"]

    def test_planned_features_heading_is_detected(self) -> None:
        text = "# X\n\n### Planned Features\n\n- something\n"
        assert checker.find_unconsolidated_future_claims(text) == ["### Planned Features"]


class TestReadmeRoadmap:
    def test_readme_has_no_stale_todos(self) -> None:
        # Happy path: shipped capabilities are not listed as to-dos.
        text = ROOT.joinpath("README.md").read_text(encoding="utf-8")
        assert checker.find_stale_readme_todos(text) == []

    @pytest.mark.parametrize("feature", checker.STALE_TODOS)
    def test_stale_todo_is_detected(self, feature: str) -> None:
        # Failure path: an already-shipped feature listed as a to-do is flagged.
        text = f"## Roadmap\n\n- [ ] {feature}\n"
        assert checker.find_stale_readme_todos(text) == [feature]

    def test_checked_item_is_not_stale(self) -> None:
        # A checked item is no longer a to-do claim of being unsupported.
        text = "## Roadmap\n\n- [x] Docker deployment\n"
        assert checker.find_stale_readme_todos(text) == []


class TestRoadmapConsolidates:
    def test_roadmap_contains_all_consolidated_claims(self) -> None:
        # Happy path: the consolidated roadmap lists every moved claim.
        text = ROOT.joinpath("docs", "ROADMAP.md").read_text(encoding="utf-8")
        assert checker.missing_roadmap_claims(text) == []

    def test_missing_claim_is_reported(self) -> None:
        # Failure path: a roadmap that drops a consolidated claim is flagged.
        text = "# Roadmap\n\n- Real-time streaming ingestion\n"
        missing = checker.missing_roadmap_claims(text)
        assert "Event Logging" in missing


class TestValidate:
    def test_clean_repo_validates(self, repo_with_roadmap: Path) -> None:
        assert checker.validate(repo_with_roadmap) == []

    def test_inline_claim_fails_validation(self, repo_with_roadmap: Path) -> None:
        (repo_with_roadmap / "docs" / "DATA_QUALITY_VALIDATION.md").write_text(
            "# X\n\n## Future Enhancements\n\n- planned\n", encoding="utf-8"
        )
        assert any("inline future-claim" in m for m in checker.validate(repo_with_roadmap))

    def test_stale_readme_fails_validation(self, repo_with_roadmap: Path) -> None:
        (repo_with_roadmap / "README.md").write_text(
            "## Roadmap\n\n- [ ] Feature store\n", encoding="utf-8"
        )
        assert any("already-shipped" in m for m in checker.validate(repo_with_roadmap))


class TestMainExitCodes:
    def test_clean_exits_zero(self) -> None:
        assert checker.main(["--root", str(ROOT), "--quiet"]) == 0

    def test_broken_exits_one(self, repo_with_roadmap: Path) -> None:
        (repo_with_roadmap / "docs" / "ROADMAP.md").write_text("# Roadmap\n", encoding="utf-8")
        assert checker.main(["--root", str(repo_with_roadmap), "--quiet"]) == 1
