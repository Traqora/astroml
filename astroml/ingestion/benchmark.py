from __future__ import annotations

"""
Ingestion benchmark utility.

Measures:
- Throughput (tx/sec) while processing a ledger range
- Memory footprint (RSS in MB) sampled at start/end
- Saves benchmark results to a JSON file for later analysis

Usage (programmatic):
  from astroml.ingestion.service import IngestionService
  from astroml.ingestion.benchmark import run_benchmark

  svc = IngestionService()
  result = run_benchmark(svc, start_ledger=0, end_ledger=999, fetch_cost_us=50)

CLI suggestion (future): expose via astroml.cli.
"""

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # Fallback to /proc/self status parsing if available

from .service import IngestionResult, IngestionService


@dataclass
class BenchmarkResult:
    start_ledger: int
    end_ledger: int
    attempted: int
    processed: int
    skipped: int
    duration_sec: float
    tx_per_sec: float
    rss_mb_start: float
    rss_mb_end: float
    rss_mb_delta: float
    timestamp: float


def _get_rss_mb() -> float:
    if psutil is not None:
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024 * 1024)
    # Fallback: read from /proc/self/statm on Linux
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
            rss_pages = int(parts[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (rss_pages * page_size) / (1024 * 1024)
    except Exception:
        return float("nan")


def run_benchmark(
    service: IngestionService,
    *,
    start_ledger: int,
    end_ledger: int,
    fetch_fn: Callable[[int], object] | None = None,
    process_fn: Callable[[int, object], None] | None = None,
    results_path: str = ".astroml_bench/ingestion_benchmark.jsonl",
    fetch_cost_us: int = 0,
    process_cost_us: int = 0,
) -> BenchmarkResult:
    """Run ingestion benchmark and persist results.

    - fetch_cost_us/process_cost_us: artificial delays (microseconds) to simulate IO/CPU costs
    - results_path: JSON lines file to append benchmark results
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    def default_fetch(ledger_id: int) -> object:
        if fetch_cost_us > 0:
            time.sleep(fetch_cost_us / 1_000_000.0)
        return {"ledger": ledger_id}

    def default_process(ledger_id: int, payload: object) -> None:
        # no-op processing; simulate CPU time if requested
        if process_cost_us > 0:
            time.sleep(process_cost_us / 1_000_000.0)
        return None

    fetch = fetch_fn or default_fetch
    process = process_fn or default_process

    rss_start = _get_rss_mb()
    t0 = time.perf_counter()
    res: IngestionResult = service.ingest(
        start_ledger=start_ledger,
        end_ledger=end_ledger,
        fetch_fn=fetch,
        process_fn=process,
    )
    t1 = time.perf_counter()
    rss_end = _get_rss_mb()

    duration = max(1e-9, t1 - t0)
    attempted = len(res.attempted)
    processed = len(res.processed)
    skipped = len(res.skipped)
    txps = attempted / duration

    bench = BenchmarkResult(
        start_ledger=start_ledger,
        end_ledger=end_ledger,
        attempted=attempted,
        processed=processed,
        skipped=skipped,
        duration_sec=duration,
        tx_per_sec=txps,
        rss_mb_start=rss_start,
        rss_mb_end=rss_end,
        rss_mb_delta=(
            (rss_end - rss_start)
            if (not (rss_start != rss_start or rss_end != rss_end))
            else float("nan")
        ),
        timestamp=time.time(),
    )

    # Persist as JSONL
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(bench)) + "\n")

    return bench


@dataclass
class ChunkedBenchmarkResult:
    """Benchmark result for the memory-optimised chunked backfill — issue #766."""

    start_ledger: int
    end_ledger: int
    chunk_size: int
    total_processed: int
    total_skipped: int
    total_errors: int
    n_chunks: int
    duration_sec: float
    tx_per_sec: float
    rss_mb_start: float
    rss_mb_end: float
    rss_mb_delta: float
    timestamp: float


def run_chunked_benchmark(
    service: IngestionService,
    start_ledger: int,
    end_ledger: int,
    chunk_size: int = 10_000,
    fetch_cost_us: int = 0,
    process_cost_us: int = 0,
    fetch_fn: Callable[[int], object] | None = None,
    process_fn: Callable[[int, object], None] | None = None,
    results_path: str = "benchmark_results/chunked_backfill.jsonl",
) -> ChunkedBenchmarkResult:
    """Benchmark memory-efficient chunked backfill for large ledger ranges.

    Processes the range in ``chunk_size`` batches via
    :meth:`~astroml.ingestion.service.IngestionService.ingest_backfill_chunked`
    and records peak-RSS at start and end so callers can compare against the
    unbounded :func:`run_benchmark` approach.

    Results are appended as JSON Lines to ``results_path`` for offline analysis.

    Args:
        service: IngestionService to benchmark.
        start_ledger: First ledger of the range (inclusive).
        end_ledger: Last ledger of the range (inclusive).
        chunk_size: Ledgers per memory-bounded batch.
        fetch_cost_us: Simulated fetch latency in microseconds.
        process_cost_us: Simulated processing latency in microseconds.
        fetch_fn: Custom fetch callable (overrides ``fetch_cost_us``).
        process_fn: Custom process callable (overrides ``process_cost_us``).
        results_path: Output file for JSON Lines results.
    """
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    def default_fetch(ledger_id: int) -> object:
        if fetch_cost_us > 0:
            time.sleep(fetch_cost_us / 1_000_000.0)
        return {"ledger": ledger_id}

    def default_process(ledger_id: int, payload: object) -> None:
        if process_cost_us > 0:
            time.sleep(process_cost_us / 1_000_000.0)

    fetch = fetch_fn or default_fetch
    process = process_fn or default_process

    rss_start = _get_rss_mb()
    t0 = time.perf_counter()

    total_processed = total_skipped = total_errors = n_chunks = 0
    for chunk in service.ingest_backfill_chunked(
        start_ledger=start_ledger,
        end_ledger=end_ledger,
        chunk_size=chunk_size,
        fetch_fn=fetch,
        process_fn=process,
    ):
        total_processed += chunk["processed"]
        total_skipped += chunk["skipped"]
        total_errors += chunk["errors"]
        n_chunks += 1

    t1 = time.perf_counter()
    rss_end = _get_rss_mb()

    duration = max(1e-9, t1 - t0)
    total_attempted = total_processed + total_skipped
    result = ChunkedBenchmarkResult(
        start_ledger=start_ledger,
        end_ledger=end_ledger,
        chunk_size=chunk_size,
        total_processed=total_processed,
        total_skipped=total_skipped,
        total_errors=total_errors,
        n_chunks=n_chunks,
        duration_sec=duration,
        tx_per_sec=total_attempted / duration,
        rss_mb_start=rss_start,
        rss_mb_end=rss_end,
        rss_mb_delta=rss_end - rss_start,
        timestamp=time.time(),
    )

    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")

    return result
