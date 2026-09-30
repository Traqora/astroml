#!/usr/bin/env python3
"""Scaffold a new documentation locale folder under docs/i18n/.

Usage:
    python docs/i18n/new_locale.py <iso-639-1-code> [--name "Native Name"]

Example:
    python docs/i18n/new_locale.py fr --name "Français"

This creates ``docs/i18n/<code>/`` and writes a ``README.<code>.md`` stub whose
section headings mirror the English ``README.md`` (prose left in English for a
contributor to translate). It is intentionally dependency-free (stdlib only) and
does not modify ``languages.yml`` — add the registry entry yourself so the change
stays explicit and reviewable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

I18N_DIR = Path(__file__).resolve().parent
REPO_ROOT = I18N_DIR.parents[1]
SOURCE_README = REPO_ROOT / "README.md"

STUB_BANNER = (
    "<!-- Partial translation. Fill in section by section; keep code blocks,\n"
    "links and identifiers identical to README.md. See docs/i18n/README.md. -->\n\n"
)


def extract_headings(md_text: str) -> list[tuple[int, str]]:
    """Return (level, text) for every ATX heading in a Markdown document.

    Headings inside fenced code blocks are ignored so shell comments such as
    ``# comment`` are not mistaken for titles.
    """
    headings: list[tuple[int, str]] = []
    in_fence = False
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped.startswith("#"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        title = stripped[level:].strip()
        if title:
            headings.append((level, title))
    return headings


def build_stub(code: str, name: str, headings: list[tuple[int, str]]) -> str:
    """Compose a README.<code>.md stub from the English heading outline."""
    lines = [f"# AstroML — {name}\n", STUB_BANNER]
    for level, title in headings:
        if level == 1 and title.lower() == "astroml":
            continue  # the localized H1 was already emitted above
        lines.append(f"{'#' * level} {title}\n")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", help="ISO 639-1 language code, e.g. 'fr'")
    parser.add_argument("--name", default=None, help="Native language name")
    args = parser.parse_args(argv)

    code = args.code.strip().lower()
    if not (len(code) == 2 and code.isalpha()):
        parser.error(f"'{args.code}' is not a 2-letter ISO 639-1 code")

    if not SOURCE_README.exists():
        print(f"error: source README not found at {SOURCE_README}", file=sys.stderr)
        return 1

    locale_dir = I18N_DIR / code
    target = locale_dir / f"README.{code}.md"
    if target.exists():
        print(f"skip: {target} already exists")
        return 0

    headings = extract_headings(SOURCE_README.read_text(encoding="utf-8"))
    stub = build_stub(code, args.name or code, headings)

    locale_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(stub, encoding="utf-8")

    print(f"created {locale_dir.relative_to(REPO_ROOT)}/")
    print(f"  wrote {target.relative_to(REPO_ROOT)} ({len(headings)} headings mirrored)")
    print("next: register the locale in docs/i18n/languages.yml, then translate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
