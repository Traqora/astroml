"""Data documentation (data docs) generation for validation results.

Resolves part of #644.

Renders expectation suites and validation results into a static site — an
index page plus one page per suite — as self-contained HTML with no external
assets, so it can be published straight to object storage or served from the
docs container.  Markdown output is available for embedding in MkDocs.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from astroml.validation.great_expectations.suite_builder import ExpectationSuite
from astroml.validation.great_expectations.validator import (
    ValidationResult,
    ValidationStore,
)

__all__ = ["DataDocsBuilder", "DataDocsPage"]

_STYLE = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 0 auto; max-width: 60rem; padding: 2rem 1rem; line-height: 1.5; }
h1, h2 { line-height: 1.2; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #8884; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #8882; }
.pass { color: #157f3b; font-weight: 600; }
.fail { color: #b3261e; font-weight: 600; }
.meta { opacity: 0.7; font-size: 0.9rem; }
.scroll { overflow-x: auto; }
code { background: #8882; padding: 0.1rem 0.3rem; border-radius: 3px; }
""".strip()


@dataclass(frozen=True)
class DataDocsPage:
    """One rendered documentation page."""

    name: str
    html: str
    markdown: str

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        """Write both renderings into ``directory`` and return their paths."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        html_path = target / f"{self.name}.html"
        md_path = target / f"{self.name}.md"
        html_path.write_text(self.html, encoding="utf-8")
        md_path.write_text(self.markdown, encoding="utf-8")
        return html_path, md_path


class DataDocsBuilder:
    """Builds a static data docs site from suites and validation results."""

    def __init__(self, output_dir: str | Path = "docs/data_docs") -> None:
        self.output_dir = Path(output_dir)

    # ── Page rendering ───────────────────────────────────────────────────────

    def suite_page(self, suite: ExpectationSuite) -> DataDocsPage:
        """Render the expectations of ``suite`` as a documentation page."""
        rows = [
            (
                expectation.expectation_type.value,
                expectation.column or "—",
                json.dumps(
                    {k: v for k, v in expectation.kwargs.items() if k != "column"},
                    default=str,
                ),
            )
            for expectation in suite.expectations
        ]
        headers = ("Expectation", "Column", "Parameters")
        title = f"Expectation suite — {suite.name}"
        body = (
            f"<p class='meta'>{len(suite.expectations)} expectations across "
            f"{len(suite.columns())} columns.</p>" + _html_table(headers, rows)
        )
        markdown = (
            f"# {title}\n\n"
            f"{len(suite.expectations)} expectations across {len(suite.columns())} columns.\n\n"
            + _markdown_table(headers, rows)
        )
        return DataDocsPage(
            name=_slugify(suite.name), html=_html_document(title, body), markdown=markdown
        )

    def result_page(self, result: ValidationResult) -> DataDocsPage:
        """Render a validation result as a documentation page."""
        rows = []
        for entry in result.results:
            status = "PASS" if entry.success else "FAIL"
            rows.append(
                (
                    status,
                    entry.expectation.expectation_type.value,
                    entry.expectation.column or "—",
                    _truncate(json.dumps(entry.observed_value, default=str)),
                    str(entry.unexpected_count),
                )
            )
        headers = ("Status", "Expectation", "Column", "Observed", "Unexpected")
        title = f"Validation — {result.suite_name} ({result.run_id[:8]})"
        state = "pass" if result.success else "fail"
        body = (
            f"<p class='{state}'>{html.escape(result.summary())}</p>"
            f"<p class='meta'>Dataset <code>{html.escape(result.dataset_name)}</code> · "
            f"run <code>{html.escape(result.run_id)}</code> · "
            f"{html.escape(result.validated_at)}</p>" + _html_table(headers, rows, status_column=0)
        )
        markdown = (
            f"# {title}\n\n"
            f"**{result.summary()}**\n\n"
            f"- Dataset: `{result.dataset_name}`\n"
            f"- Run: `{result.run_id}`\n"
            f"- Validated at: {result.validated_at}\n\n" + _markdown_table(headers, rows)
        )
        return DataDocsPage(
            name=f"{_slugify(result.suite_name)}-{result.run_id[:8]}",
            html=_html_document(title, body),
            markdown=markdown,
        )

    def index_page(self, store: ValidationStore) -> DataDocsPage:
        """Render an index of every suite with stored results."""
        rows = []
        for suite_name in store.suites():
            latest = store.latest(suite_name)
            if latest is None:
                continue
            rows.append(
                (
                    "PASS" if latest["success"] else "FAIL",
                    suite_name,
                    latest["dataset_name"],
                    f"{latest['success_percent']:.1f}%",
                    latest["validated_at"],
                )
            )
        headers = ("Status", "Suite", "Dataset", "Success", "Last run")
        title = "Data docs"
        body = "<p class='meta'>Latest validation run per expectation suite.</p>" + _html_table(
            headers, rows, status_column=0
        )
        markdown = "# Data docs\n\nLatest validation run per expectation suite.\n\n" + (
            _markdown_table(headers, rows)
        )
        return DataDocsPage(name="index", html=_html_document(title, body), markdown=markdown)

    # ── Site build ───────────────────────────────────────────────────────────

    def build(
        self,
        *,
        suites: list[ExpectationSuite] | None = None,
        results: list[ValidationResult] | None = None,
        store: ValidationStore | None = None,
    ) -> Path:
        """Write the full docs site and return its directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for suite in suites or []:
            self.suite_page(suite).write(self.output_dir / "suites")
        for result in results or []:
            self.result_page(result).write(self.output_dir / "validations")
        if store is not None:
            self.index_page(store).write(self.output_dir)
        return self.output_dir


# ─── Rendering helpers ───────────────────────────────────────────────────────


def _html_document(title: str, body: str) -> str:
    """Wrap ``body`` in a self-contained HTML document."""
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n{body}\n"
        "</body>\n</html>\n"
    )


def _html_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    status_column: int | None = None,
) -> str:
    """Render a table, colouring PASS/FAIL in ``status_column`` when given."""
    if not rows:
        return "<p class='meta'>Nothing to show.</p>"
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            escaped = html.escape(str(cell))
            if index == status_column:
                css = "pass" if cell == "PASS" else "fail"
                cells.append(f"<td class='{css}'>{escaped}</td>")
            else:
                cells.append(f"<td>{escaped}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='scroll'><table>\n<thead><tr>"
        + head
        + "</tr></thead>\n<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody>\n</table></div>"
    )


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render a Markdown table."""
    if not rows:
        return "_Nothing to show._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines) + "\n"


def _truncate(text: str, limit: int = 120) -> str:
    """Shorten ``text`` for table display."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _slugify(name: str) -> str:
    """Return a filesystem-safe slug for ``name``."""
    slug = "".join(char if char.isalnum() else "-" for char in name.lower())
    return "-".join(part for part in slug.split("-") if part) or "suite"
