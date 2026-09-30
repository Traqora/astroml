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

    agent = sub.add_parser("agent", help="LLM Agent Framework for autonomous task execution")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)

    agent_interactive = agent_sub.add_parser("interactive", help="Run agent in interactive mode")
    agent_interactive.add_argument(
        "--agent-type",
        choices=["fraud-detection", "model-training", "generic"],
        default="generic",
        help="Type of agent to use",
    )

    agent_run = agent_sub.add_parser("run", help="Run agent on a single task")
    agent_run.add_argument(
        "--agent-type",
        choices=["fraud-detection", "model-training", "generic"],
        default="generic",
        help="Type of agent to use",
    )
    agent_run.add_argument("--task", required=True, help="Task description")
    agent_run.add_argument("--context", help="Context as JSON string")
    agent_run.add_argument("--verbose", action="store_true", help="Enable verbose output")

    agent_sub.add_parser("list-tools", help="List available tools")

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
        from .agents.cli import main as agent_main

        # Build argv for agent CLI
        agent_argv = [args.agent_command]
        if args.agent_command == "interactive":
            agent_argv.extend(["--agent-type", args.agent_type])
        elif args.agent_command == "run":
            agent_argv.extend(["--agent-type", args.agent_type, "--task", args.task])
            if args.context:
                agent_argv.extend(["--context", args.context])
            if args.verbose:
                agent_argv.append("--verbose")

        return agent_main(agent_argv)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
