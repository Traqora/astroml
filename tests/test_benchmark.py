from __future__ import annotations

import os
import json

from astroml.ingestion.service import IngestionService
from astroml.ingestion.benchmark import run_benchmark


def test_benchmark_reports_and_saves(tmp_path):
    svc = IngestionService()
    outpath = tmp_path / 'bench.jsonl'

    bench = run_benchmark(
        svc,
        start_ledger=0,
        end_ledger=49,
        results_path=str(outpath),
        fetch_cost_us=0,
        process_cost_us=0,
    )

    assert bench.attempted == 50
    assert bench.tx_per_sec > 0
    # Memory fields should exist (may be NaN if platform unsupported)
    assert hasattr(bench, 'rss_mb_start') and hasattr(bench, 'rss_mb_end')

    # File exists and contains a valid JSON line
    assert outpath.exists()
    with open(outpath, 'r', encoding='utf-8') as f:
        line = f.readline()
    rec = json.loads(line)
    assert rec['attempted'] == 50
    assert 'tx_per_sec' in rec
    assert 'rss_mb_start' in rec and 'rss_mb_end' in rec
