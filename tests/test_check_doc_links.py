"""Tests for scripts/check_doc_links.py — issue #698.

The checker gates CI, so its own rules need pinning. Two of them were bugs
found while bringing the tree green, and both are regression-tested here:
the rst link pattern used to match inline-code spans, and the anchor slug
used to strip emoji before mapping spaces to hyphens.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_links.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_doc_links", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


class TestGithubSlug:
    """Anchors must match what GitHub actually generates, or good links fail."""

    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("## Getting Started", "getting-started"),
            # GitHub trims and lowercases *first*, then drops non-word chars,
            # then maps spaces to hyphens -- so the emoji leaves a leading hyphen.
            ("## 🔧 Troubleshooting", "-troubleshooting"),
            ("## What's New", "whats-new"),
            ("### API Reference", "api-reference"),
        ],
    )
    def test_slug_matches_github(self, heading: str, expected: str) -> None:
        assert checker.github_slug(heading) == expected


class TestLinkExtraction:
    def test_rst_reference_is_extracted(self) -> None:
        text = "See `the guide <guide.rst>`_ for details.\n"
        assert (1, "guide.rst") in list(checker._link_occurrences(text))

    def test_inline_code_angle_brackets_are_not_links(self) -> None:
        # `` `p <expr>` `` is an rst inline literal, not a link. Requiring the
        # trailing underscore is what separates the two.
        text = "Use `p <expr>` to filter, and `docker-compose logs <service>` too.\n"
        assert list(checker._link_occurrences(text)) == []

    def test_markdown_link_with_title(self) -> None:
        text = '[Docs](api/index.md "The API")\n'
        assert (1, "api/index.md") in list(checker._link_occurrences(text))

    def test_links_inside_fenced_blocks_are_ignored(self, tmp_path: Path) -> None:
        doc = tmp_path / "a.md"
        doc.write_text("# A\n\n```markdown\n[link](missing.md)\n```\n", encoding="utf-8")
        assert checker.scan(tmp_path, [doc], check_urls=False) == []


class TestScan:
    def test_missing_internal_target_is_reported(self, tmp_path: Path) -> None:
        doc = tmp_path / "a.md"
        doc.write_text("# A\n\n[gone](nope.md)\n", encoding="utf-8")
        broken = checker.scan(tmp_path, [doc], check_urls=False)
        assert len(broken) == 1
        assert broken[0].line == 3
        assert "nope.md" in str(broken[0])

    def test_anchor_must_exist_in_target(self, tmp_path: Path) -> None:
        (tmp_path / "b.md").write_text("# B\n\n## Real Section\n", encoding="utf-8")
        doc = tmp_path / "a.md"
        doc.write_text("[ok](b.md#real-section)\n[bad](b.md#nope-section)\n", encoding="utf-8")
        broken = checker.scan(tmp_path, [doc], check_urls=False)
        assert [b.target for b in broken] == ["b.md#nope-section"]

    def test_external_urls_skipped_without_flag(self, tmp_path: Path) -> None:
        doc = tmp_path / "a.md"
        doc.write_text("[site](https://example.invalid/does-not-exist)\n", encoding="utf-8")
        assert checker.scan(tmp_path, [doc], check_urls=False) == []


class TestMainExitCodes:
    def test_clean_tree_exits_zero(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.md").write_text("[b](b.md)\n", encoding="utf-8")
        (tmp_path / "docs" / "b.md").write_text("# B\n", encoding="utf-8")
        assert checker.main(["--root", str(tmp_path), "--quiet"]) == 0

    def test_broken_tree_exits_one(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.md").write_text("[b](b.md)\n", encoding="utf-8")
        assert checker.main(["--root", str(tmp_path), "--quiet"]) == 1

    def test_json_output_lists_findings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.md").write_text("[b](b.md)\n", encoding="utf-8")
        assert checker.main(["--root", str(tmp_path), "--json"]) == 1
        assert "b.md" in capsys.readouterr().out
