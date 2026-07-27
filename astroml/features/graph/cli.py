"""CLI for building graph snapshots with optional memory profiling — issue #546.

Usage::

    python -m astroml.features.graph.cli --memory-profile --num-edges 10000
    python -m astroml.features.graph.cli --memory-profile --source db --window 7d
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from typing import List

from astroml.features.graph.memory_profile import GraphMemoryProfile, profile_graph_memory
from astroml.features.graph.snapshot import Edge, iter_db_snapshots, window_snapshot

logger = logging.getLogger("astroml.features.graph.cli")


def _configure_logging(level: str = "INFO") -> None:
    """Configure structured logging.

    Delegates to :func:`astroml.utils.logging.configure_logging` so log level
    (``ASTROML_LOG_LEVEL``) and format stay consistent with the other astroml
    CLIs (see :mod:`astroml.ingestion.enhanced_cli`).
    """
    from astroml.utils.logging import configure_logging

    configure_logging(level=level)


def _synthetic_edges(num_edges: int, num_accounts: int, seed: int) -> List[Edge]:
    rng = random.Random(seed)
    accounts = [f"acct{i}" for i in range(max(1, num_accounts))]
    ts = 1_700_000_000
    edges = []
    for _ in range(num_edges):
        src = rng.choice(accounts)
        dst = rng.choice(accounts)
        edges.append(Edge(src=src, dst=dst, timestamp=ts))
        ts += 1
    return edges


def _run_synthetic(args: argparse.Namespace) -> GraphMemoryProfile:
    edges = _synthetic_edges(args.num_edges, args.num_accounts, args.seed)
    (nodes, windowed_edges), profile = profile_graph_memory(
        window_snapshot,
        edges,
        edges[0].timestamp,
        edges[-1].timestamp,
        n_nodes=0,
        n_edges=len(edges),
    )
    profile = GraphMemoryProfile(
        n_nodes=len(nodes),
        n_edges=len(windowed_edges),
        traced_peak_mb=profile.traced_peak_mb,
        rss_delta_mb=profile.rss_delta_mb,
        duration_s=profile.duration_s,
    )
    return profile


def _run_db(args: argparse.Namespace) -> GraphMemoryProfile:
    total_nodes = 0
    total_edges = 0

    def _build() -> None:
        nonlocal total_nodes, total_edges
        for window in iter_db_snapshots(window=args.window):
            total_nodes += len(window.nodes)
            total_edges += len(window.edges)

    _result, profile = profile_graph_memory(_build)
    return GraphMemoryProfile(
        n_nodes=total_nodes,
        n_edges=total_edges,
        traced_peak_mb=profile.traced_peak_mb,
        rss_delta_mb=profile.rss_delta_mb,
        duration_s=profile.duration_s,
    )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a graph snapshot, optionally profiling memory usage.",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Measure and print peak memory usage for the snapshot build.",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "db"],
        default="synthetic",
        help="Where to pull edges from (default: synthetic, no DB required).",
    )
    parser.add_argument(
        "--num-edges", type=int, default=10_000,
        help="Number of synthetic edges to build (--source synthetic only, default: 10000).",
    )
    parser.add_argument(
        "--num-accounts", type=int, default=500,
        help="Number of distinct synthetic accounts (--source synthetic only, default: 500).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for synthetic edge generation (default: 0).",
    )
    parser.add_argument(
        "--window", type=str, default="7d",
        help="Window size for --source db, e.g. '7d', '24h' (default: 7d).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Log level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _configure_logging(args.log_level)

    build = _run_synthetic if args.source == "synthetic" else _run_db
    profile = build(args)

    if args.memory_profile:
        print(profile.summary())
    else:
        print(f"Built snapshot: nodes={profile.n_nodes} edges={profile.n_edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
