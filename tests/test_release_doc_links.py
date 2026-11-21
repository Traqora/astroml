"""Tests for scripts/check_release_doc_links.py.

The checker gates the docs workflow, so its own rules are pinned here:
the release-notes heredoc must be locatable, its Markdown links must resolve
against the repository root, and a broken link must be reported rather than
silently shipped in a GitHub Release.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "check_release_doc_links.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_release_doc_links", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _release_yml() -> str:
    return (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


class TestExtractReleaseNotes:
    def test_real_release_notes_have_documentation_section(self) -> None:
        notes = checker.extract_release_notes(_release_yml())
        assert "## Documentation" in notes

    def test_unterminated_heredoc_raises(self) -> None:
        with pytest.raises(ValueError, match="unterminated heredoc"):
            checker.extract_release_notes("cat <<'EOF'\nbody never closes\n")

    def test_no_heredoc_raises(self) -> None:
        with pytest.raises(ValueError, match="no cat"):
            checker.extract_release_notes("nothing here\n")


class TestLinkResolution:
    def test_real_release_notes_resolve(self) -> None:
        # Happy path: every documentation link in the shipped release notes
        # points at a file that exists in the repository.
        assert checker.check_release_notes(_release_yml(), ROOT) == []

    def test_broken_link_is_reported(self, tmp_path: Path) -> None:
        # Failure path: a documentation link to a file that does not exist is
        # reported, not silently dropped.
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "ok.md").write_text("# OK\n", encoding="utf-8")
        workflow = (
            "name: Release\njobs:\n  release:\n    steps:\n"
            "      - name: Generate release notes\n        run: |\n"
            "          cat <<'EOF' > /tmp/release_notes.md\n"
            "          - [ok](docs/ok.md)\n"
            "          - [gone](docs/missing.md)\n"
            "          EOF\n"
        )
        broken = checker.check_release_notes(workflow, tmp_path)
        assert [b.target for b in broken] == ["docs/missing.md"]
        assert broken[0].kind == "missing target"

    def test_external_urls_are_skipped(self, tmp_path: Path) -> None:
        workflow = (
            "name: Release\njobs:\n  release:\n    steps:\n"
            "      - run: |\n"
            "          cat <<'EOF'\n"
            "          - [site](https://example.invalid/never)\n"
            "          EOF\n"
        )
        assert checker.check_release_notes(workflow, tmp_path) == []

    def test_links_inside_fenced_blocks_are_ignored(self, tmp_path: Path) -> None:
        workflow = (
            "name: Release\njobs:\n  release:\n    steps:\n"
            "      - run: |\n"
            "          cat <<'EOF'\n"
            "          ```bash\n"
            "          - [not a link](docs/ignored.md)\n"
            "          ```\n"
            "          EOF\n"
        )
        assert checker.check_release_notes(workflow, tmp_path) == []


class TestMainExitCodes:
    @staticmethod
    def _workflow(links: str) -> str:
        return (
            "name: Release\njobs:\n  release:\n    steps:\n"
            "      - name: Generate release notes\n        run: |\n"
            "          cat <<'EOF' > /tmp/release_notes.md\n"
            f"{links}\n"
            "          EOF\n"
        )

    def test_clean_tree_exits_zero(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "ok.md").write_text("# OK\n", encoding="utf-8")
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text(
            self._workflow("          - [ok](docs/ok.md)\n"), encoding="utf-8"
        )
        assert checker.main(["--root", str(tmp_path), "--quiet"]) == 0

    def test_broken_tree_exits_one(self, tmp_path: Path) -> None:
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "release.yml").write_text(
            self._workflow("          - [gone](docs/missing.md)\n"), encoding="utf-8"
        )
        assert checker.main(["--root", str(tmp_path), "--quiet"]) == 1

    def test_missing_workflow_exits_two(self, tmp_path: Path) -> None:
        assert checker.main(["--root", str(tmp_path)]) == 2
