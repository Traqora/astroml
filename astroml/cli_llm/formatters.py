"""Output formatting for CLI commands."""

import json
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def print_text(text: str, format: str = "text") -> None:
    if format == "markdown":
        console.print(Markdown(text))
    else:
        console.print(text)


def print_json(data: Any) -> None:
    console.print_json(json.dumps(data, default=str, indent=2))


def print_table(rows: list[list[str]], headers: list[str], title: str = "") -> None:
    table = Table(title=title, title_style="bold cyan", border_style="dim")
    for h in headers:
        table.add_column(h, style="cyan", no_wrap=True)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def print_llm_response(text: str, tokens: int = 0, cost: float = 0.0, latency: float = 0.0, model: str = "") -> None:
    console.print(Panel(text, border_style="green"))
    parts = []
    if model:
        parts.append(f"Model: {model}")
    if tokens:
        parts.append(f"Tokens: {tokens}")
    if cost:
        parts.append(f"Cost: ${cost:.5f}")
    if latency:
        parts.append(f"Latency: {latency:.2f}s")
    if parts:
        console.print(" | ".join(parts), style="dim")


def print_error(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")


def output_result(data: Any, as_json: bool = False) -> None:
    if as_json:
        print_json(data)
    else:
        console.print(str(data))
