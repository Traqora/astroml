"""Tests for scripts/check_workflow_postgres.py.

The doc ``docs/CI_DATABASE_REQUIREMENTS.md`` claims which workflows need
PostgreSQL. These tests pin the two sides of that claim against reality: the
workflows that actually start a postgres service must be the ones the doc
lists, and a workflow the doc calls database-free must stay that way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "check_workflow_postgres.py"
DOC = ROOT / "docs" / "CI_DATABASE_REQUIREMENTS.md"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_workflow_postgres", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _doc(needs: list[str], not_needs: list[str]) -> str:
    lines = [
        "# Which Workflows Need PostgreSQL",
        "",
        "### Workflows that need PostgreSQL",
        "",
        "| Workflow | Job | Postgres image | Connection string |",
        "| --- | --- | --- | --- |",
    ]
    for name in needs:
        link = f"[{name}](../.github/workflows/{name})"
        lines.append(f"| {link} | x | postgres:14 | postgresql://x |")
    lines += [
        "",
        "### Workflows that do NOT need PostgreSQL",
        "",
        "| Workflow | Job(s) | What it runs |",
        "| --- | --- | --- |",
    ]
    for name in not_needs:
        lines.append(f"| [{name}](../.github/workflows/{name}) | x | runs |")
    lines.append("")
    return "\n".join(lines)


class TestDetectPostgres:
    @pytest.mark.parametrize(
        ("yaml_text", "expected"),
        [
            ("services:\n  postgres:\n    image: postgres:14\n", True),
            ("env:\n  DATABASE_URL: postgresql://u:p@h:5432/db\n", True),
            ("services:\n  redis:\n    image: redis:7\n", False),
            ("run: kubectl rollout status deployment/postgres -n astroml\n", False),
            ("name: Lint\njobs:\n  lint:\n    runs-on: ubuntu-latest\n", False),
        ],
        ids=[
            "postgres-service",
            "postgresql-url",
            "redis-only",
            "kubectl-deploy",
            "plain",
        ],
    )
    def test_detection(self, yaml_text: str, expected: bool) -> None:
        assert checker.detect_postgres(yaml_text) is expected


class TestValidate:
    def test_real_doc_matches_workflows(self) -> None:
        # Happy path: the shipped doc agrees with the real workflow files.
        assert checker.validate(ROOT, DOC.read_text(encoding="utf-8")) == []

    def test_real_doc_lists_the_postgres_workflows(self) -> None:
        # Explicit: the three workflows that start postgres are exactly the
        # ones the doc lists as needing it.
        detected = checker.workflows_needing_postgres(ROOT)
        assert detected == checker.doc_needs_postgres(DOC.read_text(encoding="utf-8"))
        assert detected == {"ci.yml", "e2e-tests.yml", "nightly.yml"}

    def test_false_positive_is_flagged(self) -> None:
        # Failure path: a workflow listed as needing PostgreSQL that does not
        # use it is reported.
        doc = _doc(
            needs=["ci.yml", "e2e-tests.yml", "nightly.yml", "docs.yml"],
            not_needs=["smoke-tests.yml"],
        )
        mismatches = checker.validate(ROOT, doc)
        assert any("docs.yml" in m for m in mismatches)

    def test_false_negative_is_flagged(self) -> None:
        # Failure path: a workflow that uses PostgreSQL but is omitted from the
        # doc is reported.
        doc = _doc(needs=["ci.yml", "e2e-tests.yml"], not_needs=["nightly.yml"])
        mismatches = checker.validate(ROOT, doc)
        assert any("nightly.yml" in m and "uses PostgreSQL" in m for m in mismatches)

    def test_contradiction_is_flagged(self) -> None:
        # Failure path: a workflow listed as NOT needing PostgreSQL that does
        # use it is reported.
        doc = _doc(needs=["ci.yml", "e2e-tests.yml"], not_needs=["nightly.yml"])
        mismatches = checker.validate(ROOT, doc)
        assert any("nightly.yml" in m and "NOT needing" in m for m in mismatches)

    def test_missing_table_is_reported(self) -> None:
        doc = "# Which Workflows Need PostgreSQL\n\n### Workflows that need PostgreSQL\n\n| W |\n"
        mismatches = checker.validate(ROOT, doc)
        assert mismatches and "could not parse" in mismatches[0]


class TestMainExitCodes:
    def test_clean_exits_zero(self) -> None:
        assert checker.main(["--root", str(ROOT), "--quiet"]) == 0

    def test_broken_doc_exits_one(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir(parents=True)
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        # A real postgres workflow so detection finds something, plus a doc
        # that omits it -> mismatch -> exit 1.
        (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
            "services:\n  postgres:\n    image: postgres:14\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "CI_DATABASE_REQUIREMENTS.md").write_text(_doc([], []), encoding="utf-8")
        assert checker.main(["--root", str(tmp_path), "--quiet"]) == 1

    def test_missing_doc_exits_two(self, tmp_path: Path) -> None:
        assert checker.main(["--root", str(tmp_path)]) == 2
