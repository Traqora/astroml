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

from dataclasses import dataclass, asdict
import json
import os
import time
from typing import Callable, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # Fallback to /proc/self status parsing if available

from .service import IngestionService, IngestionResult


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
        with open('/proc/self/statm', 'r') as f:
            parts = f.read().split()
            rss_pages = int(parts[1])
        page_size = os.sysconf('SC_PAGE_SIZE')
        return (rss_pages * page_size) / (1024 * 1024)
    except Exception:
        return float('nan')


def run_benchmark(
    service: IngestionService,
    *,
    start_ledger: int,
    end_ledger: int,
    fetch_fn: Optional[Callable[[int], object]] = None,
    process_fn: Optional[Callable[[int, object], None]] = None,
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
        rss_mb_delta=(rss_end - rss_start) if (not (rss_start != rss_start or rss_end != rss_end)) else float('nan'),
        timestamp=time.time(),
    )

    # Persist as JSONL
    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(asdict(bench)) + "\n")

    return bench
