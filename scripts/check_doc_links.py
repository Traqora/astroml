#!/usr/bin/env python3
"""Check the internal links in the documentation.

Issue #698. The docs tree is a mix of Markdown and reStructuredText, and both
formats accept relative links that a refactor silently orphans: a page renamed,
a section reworded, a path that only made sense from the repository root. Sphinx
does not catch any of it for the Markdown pages, because ``docs/conf.py`` has
never enabled a Markdown parser -- so those 30-odd files are not built at all
and nothing in CI ever reads them.

This script is the missing check. It resolves every *internal* link it can find
and reports the ones that do not land. External URLs are only fetched with
``--check-urls``, because a documentation build should not fail because a
third-party site had a bad afternoon.

Usage::

    python scripts/check_doc_links.py                 # check docs/ and README
    python scripts/check_doc_links.py --check-urls    # also fetch http(s) links
    python scripts/check_doc_links.py --json          # machine-readable output

Exit status is 0 when every link resolves and 1 otherwise, so it drops straight
into CI and pre-commit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Directories never scanned, whichever format they are in.
SKIP_DIRS = frozenset({"_build", "_static", "_templates", "node_modules", ".venv", "venv", ".git"})

DOC_SUFFIXES = (".md", ".rst")

EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "git:", "ftp://")

#: Markdown inline link: ``[label](target)`` with an optional ``"title"``.
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(\s*([^()\s]+)(?:\s+\"[^\"]*\")?\s*\)")

#: reStructuredText named hyperlink target: ``` `label <target>`_ ```. The
#: trailing underscore is what separates a link from an inline-literal span
#: such as ``p <expr>``, so it is required rather than optional here.
_RST_LINK = re.compile(r"(?<![:\w])`[^`]+?\s<([^>\s]+)>`_(?!_)")

#: Fenced code block delimiters; links inside them are literal text, not links.
_FENCE = re.compile(r"^\s*(```|~~~)")

#: rst directives and roles that look like markup but carry no link.
_CODE_BLOCK_MARKER = re.compile(r"^\s*(?:\.\. )?code-block::|^::\s*$")


@dataclass(frozen=True)
class BrokenLink:
    """One link that does not resolve."""

    source: str
    line: int
    target: str
    kind: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "line": self.line,
            "target": self.target,
            "kind": self.kind,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        return f"{self.source}:{self.line}: {self.target} -- {self.kind}: {self.detail}"


def github_slug(heading: str) -> str:
    """The anchor GitHub generates for a Markdown heading.

    GitHub trims and lowercases *first*, then drops characters that are not
    word characters, spaces or hyphens, then maps each remaining space to a
    hyphen. The order is what makes an emoji heading such as
    ``## 🔧 Troubleshooting`` slug to ``-troubleshooting`` with a leading
    hyphen: the trim happens while the emoji is still a character occupying
    the start of the line, so the space behind it survives to become one.
    """
    text = heading.strip().lstrip("#").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def iter_docs(root: Path, paths: Iterable[str]) -> Iterator[Path]:
    """Yield every documentation file under ``root`` (or under ``paths``)."""
    if paths:
        for raw in paths:
            candidate = (root / raw).resolve()
            if candidate.is_file() and candidate.suffix in DOC_SUFFIXES:
                yield candidate
            elif candidate.is_dir():
                yield from iter_docs(candidate, [])
        return
    for path in sorted(root.rglob("*")):
        if path.suffix not in DOC_SUFFIXES or not path.is_file():
            continue
        if SKIP_DIRS.intersection(path.parts):
            continue
        yield path


def _anchor_lines(text: str, is_markdown: bool) -> set[str]:
    """Collect anchors a document can be linked to."""
    anchors: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if is_markdown:
            if line.startswith("#"):
                anchors.add(github_slug(line))
                # A leading HTML anchor, ``<a id="foo"></a>``, is a common way
                # to pin a stable anchor over a rewordable heading.
                anchors.update(re.findall(r'<a\s+id=["\']([^"\']+)["\']', line))
        else:
            # rst labels: ``.. _name:`` (explicit), and section titles get an
            # implicit ``<section>`` label we cannot cheaply derive.
            match = re.match(r"^\s*\.\.\s+_([^:]+):\s*$", line)
            if match:
                anchors.add(match.group(1))
    return anchors


def _link_occurrences(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, target)`` for every inline link in a document.

    Fenced code blocks and literal ``code-block`` regions are skipped: a link
    there is sample output, not a navigable reference.
    """
    lines = text.splitlines()
    in_fence = False
    skip_until_dedent = False
    for number, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if _CODE_BLOCK_MARKER.match(line):
            skip_until_dedent = True
            continue
        if skip_until_dedent:
            # rst/code-block bodies are indented; anything at column zero ends it.
            if line.strip() and not line.startswith((" ", "\t")):
                skip_until_dedent = False
            else:
                continue
        for pattern in (_MARKDOWN_LINK, _RST_LINK):
            for match in pattern.finditer(line):
                target = match.group(1)
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                yield number, target


def check_link(root: Path, source: Path, target: str) -> tuple[str, str] | None:
    """Return ``(kind, detail)`` when ``target`` does not resolve, else ``None``."""
    if target.startswith(EXTERNAL_SCHEMES):
        return None

    path_part, _, anchor = target.partition("#")

    if path_part.startswith("/"):
        resolved = root / path_part.lstrip("/")
    elif path_part:
        resolved = (source.parent / path_part).resolve()
    else:
        resolved = source

    if not resolved.exists():
        return ("missing target", f"no such file or directory: {resolved.relative_to(root)}")

    if not anchor:
        return None

    if resolved.suffix not in DOC_SUFFIXES:
        # Anchors into source files cannot be verified without a parser per
        # language; the path above is the meaningful check there.
        return None

    anchors = _anchor_lines(
        resolved.read_text(encoding="utf-8", errors="replace"), resolved.suffix == ".md"
    )
    if anchor in anchors:
        return None
    return ("missing anchor", f"{resolved.relative_to(root)} has no anchor '{anchor}'")


def check_url(target: str, timeout: float = 10.0) -> tuple[str, str] | None:
    """Fetch an external URL and report a failure."""
    request = urllib.request.Request(  # noqa: S310 - targets come from repo docs
        target,
        headers={"User-Agent": "astroml-doc-link-check"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            code = response.status
    except urllib.error.HTTPError as exc:
        # 405/403 usually means "HEAD not allowed", which is not a dead link.
        if exc.code in (401, 403, 405):
            return None
        return ("unreachable", f"HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001 - any network failure is reported, not fatal
        return ("unreachable", type(exc).__name__)
    if code >= 400:
        return ("unreachable", f"HTTP {code}")
    return None


def scan(root: Path, files: Iterable[Path], check_urls: bool) -> list[BrokenLink]:
    broken: list[BrokenLink] = []
    url_cache: dict[str, tuple[str, str] | None] = {}
    for document in files:
        try:
            text = document.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - unreadable file is itself a report
            broken.append(
                BrokenLink(str(document.relative_to(root)), 0, "", "unreadable", str(exc))
            )
            continue
        rel = str(document.relative_to(root))
        for line, target in _link_occurrences(text):
            problem = check_link(root, document, target)
            kind = "internal"
            if problem is None and check_urls and target.startswith(EXTERNAL_SCHEMES):
                kind = "external"
                if target not in url_cache:
                    url_cache[target] = check_url(target)
                problem = url_cache[target]
            if problem is not None:
                broken.append(BrokenLink(rel, line, target, f"{kind} {problem[0]}", problem[1]))
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="*", help="files or directories to scan (default: docs/ + README)"
    )
    parser.add_argument(
        "--root", default=".", help="repository root used to resolve absolute links"
    )
    parser.add_argument(
        "--check-urls", action="store_true", help="also fetch external http(s) links"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--quiet", action="store_true", help="only print the summary line")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if args.paths:
        targets = list(iter_docs(root, args.paths))
    else:
        targets = sorted(
            set(iter_docs(root / "docs", []))
            | set(iter_docs(root, ["README.md", "CONTRIBUTING.md"]))
        )

    broken = scan(root, targets, args.check_urls)

    if args.json:
        print(json.dumps([link.as_dict() for link in broken], indent=2))
    else:
        if broken and not args.quiet:
            for link in broken:
                print(link)
            print()
        summary = (
            f"checked {len(targets)} document(s): {len(broken)} broken link(s)"
            if broken
            else f"checked {len(targets)} document(s): all links resolve"
        )
        print(summary)

    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
