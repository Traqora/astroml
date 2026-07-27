from __future__ import annotations

import argparse
import json
from typing import Optional

from .ingestion.service import IngestionService
from .ingestion.state import StateStore


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="astroml", description="AstroML utilities CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Incremental ingestion of ledgers")
    ingest.add_argument("--start", type=int, default=None, help="Start ledger id (inclusive)")
    ingest.add_argument("--end", type=int, default=None, help="End ledger id (inclusive)")
    ingest.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to state file (defaults to ./.astroml_state/ingestion_state.json)",
    )

    preprocess = sub.add_parser(
        "preprocess-backfill",
        help="Preprocess large ledger backfill datasets using Polars",
    )
    preprocess.add_argument(
        "--input",
        required=True,
        help="Input file or directory (csv, parquet, ndjson/jsonl).",
    )
    preprocess.add_argument(
        "--output",
        required=True,
        help="Output Parquet path.",
    )
    preprocess.add_argument(
        "--input-format",
        choices=["parquet", "csv", "ndjson", "jsonl"],
        default=None,
        help="Optional explicit input format.",
    )

    # Agent subcommand
    agent = sub.add_parser("agent", help="LLM Agent Framework — multi-step reasoning and autonomous task execution")
    agent.add_argument("agent_command", choices=["run", "tools"], help="Agent subcommand")
    agent.add_argument("task", nargs="?", default=None, help="Task description (for 'run')")
    agent.add_argument("--agent-type", choices=["react", "cot", "planner"], default="planner", help="Agent type")
    agent.add_argument("--provider", choices=["mock", "openai", "anthropic"], default="mock", help="LLM provider")
    agent.add_argument("--model", default="gpt-3.5-turbo", help="Model name")
    agent.add_argument("--api-key", default=None, help="API key")
    agent.add_argument("--max-steps", type=int, default=50, help="Max reasoning steps")
    agent.add_argument("--verbose", action="store_true", help="Verbose output")
    agent.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        store = StateStore(path=args.state_file) if args.state_file else StateStore()
        service = IngestionService(state_store=store)

        # Example fetch/process functions; in real usage, users would customize/import
        def fetch_fn(ledger_id: int):
            # Placeholder fetch, replace with real data retrieval
            return {"ledger": ledger_id, "data": f"payload-{ledger_id}"}

        def process_fn(ledger_id: int, payload: dict):
            # Placeholder processing; replace with DB writes or other side effects
            # For CLI visibility we do minimal printing; real apps would use logging
            print(f"processed ledger {ledger_id}")

        result = service.ingest(
            start_ledger=args.start,
            end_ledger=args.end,
            fetch_fn=fetch_fn,
            process_fn=process_fn,
        )
        print(json.dumps({
            "attempted": result.attempted,
            "processed": result.processed,
            "skipped": result.skipped,
        }, indent=2))
        return 0

    if args.command == "preprocess-backfill":
        from .preprocessing.ledger_backfill import preprocess_to_parquet

        output_path = preprocess_to_parquet(
            input_path=args.input,
            output_path=args.output,
            input_format=args.input_format,
        )
        print(json.dumps({"output": str(output_path)}, indent=2))
        return 0

    if args.command == "agent":
        from .agent.config import AgentConfig, LLMConfig, MemoryConfig, ExecutorConfig
        from .agent.executor import AutonomousExecutor
        from .agent.tools import create_default_registry

        if args.agent_command == "tools":
            registry = create_default_registry()
            print(json.dumps({"tools": sorted(registry.list_tools())}, indent=2))
            return 0

        # agent run
        llm_config = LLMConfig(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
        )
        agent_config = AgentConfig(
            llm=llm_config,
            agent_type=args.agent_type,
            verbose=args.verbose,
        )
        executor = AutonomousExecutor(agent_config=agent_config)
        result = executor.run(task_description=args.task or "")

        if args.json:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        elif args.verbose:
            print(json.dumps({
                "task": result.task,
                "success": result.success,
                "output": result.output,
                "elapsed_seconds": result.elapsed_seconds,
                "steps": len(result.steps),
                "error": result.error,
            }, indent=2))
        else:
            print(result.output)

        return 0 if result.success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
