"""Command-line entry point for the LLM Agent Framework.

Usage::

    python -m astroml.agent run "Calculate the area of a circle with radius 5"
    python -m astroml.agent run --agent-type react "List files in the current directory"
    python -m astroml.agent run --verbose "Explain the concept of dynamic graph learning"
    python -m astroml.agent interactive
    python -m astroml.agent tools
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .config import AgentConfig
from .executor import AutonomousExecutor
from .tools import create_default_registry


def _build_config(args: argparse.Namespace) -> AgentConfig:
    """Build an :class:`AgentConfig` from CLI arguments."""
    from .config import LLMConfig, MemoryConfig, ExecutorConfig

    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
    )
    mem_config = MemoryConfig(
        short_term_limit=args.memory_limit,
    )
    exec_config = ExecutorConfig(
        max_steps=args.max_steps,
        task_timeout=args.task_timeout,
        auto_approve_tools=args.auto_approve,
    )
    return AgentConfig(
        llm=llm_config,
        memory=mem_config,
        executor=exec_config,
        agent_type=args.agent_type,
        name=args.name,
        verbose=args.verbose,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run a task autonomously."""
    config = _build_config(args)
    executor = AutonomousExecutor(agent_config=config)

    result = executor.run(
        task_description=args.task,
        success_criteria=args.success_criteria,
        constraints=args.constraints,
        context=args.context,
        timeout=args.timeout,
    )

    if args.verbose or args.json:
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            print(f"\n{'=' * 60}")
            print(f"Task: {result.task}")
            print(f"Success: {result.success}")
            print(f"Time: {result.elapsed_seconds:.2f}s")
            print(f"Steps: {len(result.steps)}")
            if result.plan:
                print(f"Plan: {len(result.plan.steps)} steps")
            print(f"{'=' * 60}")
            print(f"\nOutput:\n{result.output}")
            if result.error:
                print(f"\nError: {result.error}")
    else:
        print(result.output)

    return 0 if result.success else 1


def cmd_interactive(args: argparse.Namespace) -> int:
    """Start an interactive agent session."""
    config = _build_config(args)
    executor = AutonomousExecutor(agent_config=config)

    print(f"🤖 {config.name} — Interactive Agent Session")
    print(f"Agent type: {config.agent_type} | Provider: {config.llm.provider}")
    print("Type 'quit' or 'exit' to end.\n")

    while True:
        try:
            task = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if task.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            return 0

        if not task:
            continue

        print("🤖 Thinking...")
        result = executor.run(task_description=task)
        print(f"🤖 {result.output}\n")

    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """List available tools."""
    registry = create_default_registry()
    print("Available tools:")
    print()
    for name in sorted(registry.list_tools()):
        tool = registry.get(name)
        print(f"  {name}")
        print(f"    {tool.description}")
        print()
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Run multiple tasks from a file."""
    config = _build_config(args)
    executor = AutonomousExecutor(agent_config=config)

    with open(args.tasks_file, "r", encoding="utf-8") as f:
        tasks = [line.strip() for line in f if line.strip()]

    results = executor.run_batch(tasks)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
    else:
        print(executor.summarize(results))

    return 0 if all(r.success for r in results) else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="astroml.agent",
        description="LLM Agent Framework for AstroML — multi-step reasoning and autonomous task execution.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--agent-type", choices=["react", "cot", "planner"], default="planner",
                        help="Agent type (default: planner)")
    common.add_argument("--provider", choices=["mock", "openai", "anthropic"], default="mock",
                        help="LLM provider (default: mock)")
    common.add_argument("--model", default="gpt-3.5-turbo", help="Model name")
    common.add_argument("--api-key", default=None, help="API key (or set env var)")
    common.add_argument("--api-base", default=None, help="Custom API base URL")
    common.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response")
    common.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    common.add_argument("--timeout", type=float, default=60.0, help="Request timeout (seconds)")
    common.add_argument("--max-steps", type=int, default=50, help="Max reasoning steps")
    common.add_argument("--task-timeout", type=float, default=600.0, help="Task timeout (seconds)")
    common.add_argument("--memory-limit", type=int, default=20, help="Short-term memory limit")
    common.add_argument("--auto-approve", action="store_true", default=True,
                        help="Auto-approve tool execution")
    common.add_argument("--name", default="astroml-agent", help="Agent name")
    common.add_argument("--verbose", action="store_true", help="Verbose output")

    # run
    run = sub.add_parser("run", parents=[common], help="Run a task autonomously")
    run.add_argument("task", help="The task description")
    run.add_argument("--success-criteria", nargs="*", default=[],
                     help="Success criteria (list of strings)")
    run.add_argument("--constraints", nargs="*", default=[],
                     help="Constraints (list of strings)")
    run.add_argument("--context", default="", help="Additional context")
    run.add_argument("--json", action="store_true", help="Output as JSON")
    run.set_defaults(func=cmd_run)

    # interactive
    interactive = sub.add_parser("interactive", parents=[common], help="Start an interactive session")
    interactive.set_defaults(func=cmd_interactive)

    # tools
    tools = sub.add_parser("tools", parents=[common], help="List available tools")
    tools.set_defaults(func=cmd_tools)

    # batch
    batch = sub.add_parser("batch", parents=[common], help="Run tasks from a file")
    batch.add_argument("tasks_file", help="File with one task per line")
    batch.add_argument("--json", action="store_true", help="Output as JSON")
    batch.set_defaults(func=cmd_batch)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
