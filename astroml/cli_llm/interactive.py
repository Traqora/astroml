"""Interactive chat mode with streaming support."""

import sys
from typing import Any

from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from astroml.llm.providers.base import LLMProvider
from .formatters import console


def run_chat(
    provider: LLMProvider,
    model: str = "",
    system_prompt: str = "",
    **kwargs: Any,
) -> None:
    """Run an interactive chat session with streaming."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    console.print("[bold cyan]Interactive Chat Mode[/bold cyan]")
    console.print("Type [dim]/exit[/dim] to quit, [dim]/clear[/dim] to clear history")
    console.print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/clear":
            messages.clear()
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            console.print("[dim]History cleared[/dim]")
            continue

        messages.append({"role": "user", "content": user_input})

        prompt_text = "\n".join(m["content"] for m in messages if m["content"])

        collected = ""
        with Live(Text("▌"), refresh_per_second=20) as live:
            try:
                for chunk in provider.stream(prompt_text, model=model or None, **kwargs):
                    collected += chunk
                    live.update(Text(collected + "▌"))
            except Exception as e:
                live.update(Text(f"[Error: {e}]", style="red"))
                break

        messages.append({"role": "assistant", "content": collected})
        console.print()
