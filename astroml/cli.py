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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
