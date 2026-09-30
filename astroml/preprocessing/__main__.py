"""CLI for Polars-based ledger backfill preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from astroml.preprocessing.ledger_backfill import preprocess_to_parquet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m astroml.preprocessing",
        description="Preprocess ledger backfill datasets with Polars",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory (csv, parquet, ndjson/jsonl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--input-format",
        choices=["parquet", "csv", "ndjson", "jsonl"],
        default=None,
        help="Optional explicit format. If omitted, inferred from input path.",
    )
    args = parser.parse_args(argv)

    output = preprocess_to_parquet(
        input_path=Path(args.input),
        output_path=Path(args.output),
        input_format=args.input_format,
    )
    print(f"Wrote preprocessed dataset to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
