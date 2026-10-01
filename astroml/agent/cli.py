"""Command line interface for the AstroML agent framework.

Examples::

    # Offline smoke test with the deterministic echo provider
    python -m astroml.agent "Summarise the risk posture of this graph"

    # Against a local Ollama server
    python -m astroml.agent --provider ollama --model llama3.1 \\
        "Rank the busiest accounts"

    # Load a graph from JSON and print the full trace as JSON
    python -m astroml.agent --edges data/graph.json --json \\
        "How many accounts are there?"

Provider settings fall back to the ``ASTROML_LLM_*`` environment variables
when no CLI override is supplied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import domain_tools
from .compression import CompressionConfig, PromptCompressor
from .domain_tools import build_default_registry
from .executor import AgentExecutor
from .llm import LLMProvider, provider_from_env
from .planner import TaskPlanner
from .tools import Tool, ToolError, ToolRegistry, tool_from_callable
from .types import AgentConfig, AgentRunResult, AgentStep

PROVIDER_CHOICES = (
    "echo",
    "scripted",
    "openai",
    "openai-compatible",
    "ollama",
    "vllm",
    "lmstudio",
)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the ``astroml agent`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="astroml agent",
        description="Run the AstroML LLM agent on a goal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("goal", nargs="?", help="Goal for the agent to accomplish")
    parser.add_argument(
        "--goal",
        dest="goal_flag",
        default=None,
        help="Alternative to the positional goal argument",
    )
    parser.add_argument(
        "--edges",
        default=None,
        help=(
            "JSON file containing a list of transaction edges (or "
            "{'edges': [...]}). When supplied, the graph tools operate on "
            "this dataset directly."
        ),
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=PROVIDER_CHOICES,
        help="LLM provider backend (default: ASTROML_LLM_PROVIDER or 'echo')",
    )
    parser.add_argument("--model", default=None, help="Model identifier")
    parser.add_argument("--base-url", default=None, help="API root URL")
    parser.add_argument("--api-key", default=None, help="Bearer token, if needed")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "native", "react"],
        help="Reasoning strategy: auto (default), native tool calling or ReAct",
    )
    parser.add_argument("--max-steps", type=int, default=8, help="Step budget")
    parser.add_argument(
        "--max-tool-errors",
        type=int,
        default=3,
        help="Abort after this many failed tool calls",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Pre-decompose the goal with TaskPlanner before execution",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help="Comma separated allow-list of tool names",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full run trace as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step progress and the trace summary",
    )
    compression = parser.add_argument_group("prompt compression")
    compression.add_argument(
        "--compress",
        action="store_true",
        help=(
            "Compress the prompt sent to the model each turn (whitespace, tool "
            "output, duplicate observations, digest of old turns)"
        ),
    )
    compression.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Token budget per model call; trimming is applied until the prompt "
            "fits. Implies --compress."
        ),
    )
    compression.add_argument(
        "--compress-keep-recent",
        type=int,
        default=None,
        metavar="N",
        help="Most recent turns pinned by compression (default: 6). Implies --compress.",
    )
    compression.add_argument(
        "--tool-output-limit",
        type=int,
        default=None,
        metavar="N",
        help="Token ceiling per tool observation (default: 800). Implies --compress.",
    )
    compression.add_argument(
        "--compact-tools",
        action="store_true",
        help="Render the tool catalogue one line per tool to save prompt tokens",
    )
    return parser


def _build_provider(args: argparse.Namespace) -> LLMProvider:
    """Resolve the provider from CLI flags, falling back to the environment."""
    overrides: Dict[str, str] = {}
    if args.provider:
        overrides["ASTROML_LLM_PROVIDER"] = args.provider
    if args.model:
        overrides["ASTROML_LLM_MODEL"] = args.model
    if args.base_url:
        overrides["ASTROML_LLM_BASE_URL"] = args.base_url
    if args.api_key:
        overrides["ASTROML_LLM_API_KEY"] = args.api_key

    if not overrides:
        return provider_from_env()

    env = dict(os.environ)
    env.update(overrides)
    return provider_from_env(env)


def _load_edges(path: str) -> List[Dict[str, Any]]:
    """Load an edge list from a JSON file.

    Reads with ``utf-8-sig`` so files written by Windows tooling (which often
    prepend a UTF-8 BOM) load without manual cleanup.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        if "edges" not in data:
            raise ValueError(
                "edges file must contain a JSON list or an object with an "
                "'edges' key"
            )
        data = data["edges"]
    if not isinstance(data, list):
        raise ValueError("edges file must contain a JSON list or {'edges': [...]}")

    edges = [dict(item) for item in data if isinstance(item, dict)]
    if data and not edges:
        raise ValueError("edges file must contain objects with src/dst/timestamp")
    return edges


def _bound_graph_tools(edges: List[Dict[str, Any]]) -> List[Tool]:
    """Tools bound to an already-loaded edge list.

    Binding keeps the dataset out of the prompt: the model only supplies
    scalar arguments, so graphs of any size can be analysed.
    """

    def _reference_time() -> float:
        timestamps = [
            float(edge.get("timestamp", 0) or 0)
            for edge in edges
            if isinstance(edge, dict)
        ]
        return max(timestamps) if timestamps else 0.0

    def graph_overview() -> Dict[str, Any]:
        """Summarise the loaded graph: size, assets, density, busiest accounts."""
        return domain_tools.graph_overview(edges)

    def window_stats(start_ts: int, end_ts: int) -> Dict[str, Any]:
        """Count nodes and edges in an inclusive [start_ts, end_ts] window."""
        return domain_tools.window_stats(edges, start_ts, end_ts)

    def account_features(
        accounts: Optional[List[str]] = None,
        ref_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute per-account features (degrees, volume, age, asset diversity)."""
        reference = float(ref_time) if ref_time is not None else _reference_time()
        return domain_tools.account_features(edges, reference, accounts)

    def top_accounts(
        metric: str = "volume",
        top_k: int = domain_tools.DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Rank accounts by volume, degree, in_degree or out_degree."""
        return domain_tools.top_accounts(edges, metric=metric, top_k=top_k)

    def asset_breakdown() -> Dict[str, int]:
        """Count the loaded edges per asset symbol, most frequent first."""
        counts: Dict[str, int] = {}
        for edge in edges:
            asset = edge.get("asset") if isinstance(edge, dict) else None
            if asset is None:
                continue
            key = str(asset)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    def sample_edges(limit: int = 5) -> List[Dict[str, Any]]:
        """Return a small sample of the loaded edges so you can inspect the schema."""
        return edges[: max(1, int(limit))]

    return [
        tool_from_callable(graph_overview),
        tool_from_callable(window_stats),
        tool_from_callable(account_features),
        tool_from_callable(top_accounts),
        tool_from_callable(asset_breakdown),
        tool_from_callable(sample_edges),
    ]


def _short(text: Optional[str], limit: int = 160) -> str:
    """Collapse whitespace and truncate text for terminal output."""
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _print_step(step: AgentStep) -> None:
    """Progress callback that writes each completed step to stderr."""
    sys.stderr.write(f"[step {step.index}] {step.status.value}\n")
    if step.thought:
        sys.stderr.write(f"  thought: {_short(step.thought)}\n")
    for call in step.tool_calls:
        rendered = json.dumps(call.arguments, default=str)
        sys.stderr.write(f"  -> {call.name}({_short(rendered)})\n")
    for result in step.results:
        marker = "ok" if result.ok else "error"
        sys.stderr.write(f"  <- {marker}: {_short(result.content)}\n")


def _build_compressor(args: argparse.Namespace) -> Optional[PromptCompressor]:
    """Build the CLI's compressor, or ``None`` when compression is disabled.

    Every budget flag implies ``--compress``, so ``--max-prompt-tokens 4000``
    on its own does what it says.  Unspecified tunables fall back to the
    :class:`CompressionConfig` defaults.
    """
    enabled = (
        args.compress
        or args.max_prompt_tokens is not None
        or args.compress_keep_recent is not None
        or args.tool_output_limit is not None
    )
    if not enabled:
        return None

    overrides: Dict[str, Any] = {}
    if args.max_prompt_tokens is not None:
        overrides["max_prompt_tokens"] = args.max_prompt_tokens
    if args.compress_keep_recent is not None:
        overrides["keep_recent"] = args.compress_keep_recent
    if args.tool_output_limit is not None:
        overrides["tool_output_limit"] = args.tool_output_limit
    return PromptCompressor(config=CompressionConfig(**overrides))


def _print_compression(result: AgentRunResult) -> None:
    """Write the run's aggregate prompt-compression summary to stderr."""
    data = result.trace.metadata.get("compression") or {}
    if not data:
        return
    before = int(data.get("before_tokens", 0))
    after = int(data.get("after_tokens", 0))
    saved = max(0, before - after)
    ratio = (saved / before) if before else 0.0
    print(
        f"compression: {before} -> {after} prompt tokens (-{ratio:.1%}) "
        f"over {int(data.get('turns', 0))} turn(s)",
        file=sys.stderr,
    )
    print(
        f"  strategies: {', '.join(data.get('strategies', [])) or 'none'}",
        file=sys.stderr,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``astroml agent`` and ``python -m astroml.agent``."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    goal = (args.goal or args.goal_flag or "").strip()
    if not goal:
        parser.error("a goal is required (positional or --goal)")

    tool_names: Optional[List[str]] = None
    if args.tools:
        tool_names = [name.strip() for name in args.tools.split(",") if name.strip()]

    try:
        provider = _build_provider(args)
        config = AgentConfig(
            max_steps=args.max_steps,
            max_tool_errors=args.max_tool_errors,
            mode=args.mode,
            compact_tool_catalogue=args.compact_tools,
        )

        if args.edges:
            registry = ToolRegistry(_bound_graph_tools(_load_edges(args.edges)))
        else:
            registry = build_default_registry()

        if tool_names:
            registry = registry.filtered(tool_names)

        agent = AgentExecutor(
            llm=provider,
            tools=registry,
            config=config,
            planner=(
                TaskPlanner(provider, max_steps=config.max_steps)
                if args.plan
                else None
            ),
            on_step=None if args.quiet else _print_step,
            compressor=_build_compressor(args),
        )
        result = agent.run(goal)
    except (KeyError, OSError, RuntimeError, ToolError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(result.answer if result.answer is not None else "(no answer)")
        if not args.quiet:
            print("-" * 60, file=sys.stderr)
            print(result.trace.summary(), file=sys.stderr)
            _print_compression(result)

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
