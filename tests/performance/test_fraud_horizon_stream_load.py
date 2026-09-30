"""p95 load test for the Horizon streaming client's fraud ingestion path (issue #959).

Fraud scoring (``astroml/api/routers/fraud.py``) depends on transactions
reaching handlers promptly and exactly once as they arrive off the Horizon
SSE stream (``astroml.ingestion.horizon_stream.HorizonStreamingClient``).
Under a burst of ledger activity — e.g. catching up after a reconnect, or a
network-wide spike in transaction volume — per-transaction handling latency
must stay bounded so a downstream fraud scorer isn't starved of fresh data.

This benchmarks ``HorizonStreamingClient._handle_payload``, the per-frame
decode -> de-duplicate -> dispatch hot path shared by every transaction the
client delivers, and asserts a p95 latency budget under sustained load.
Complements the correctness-focused idempotency suite in
``tests/test_horizon_stream_idempotency.py``, which covers replay
suppression but not throughput/latency under load.
"""

from __future__ import annotations

import asyncio
import json
import statistics

import pytest

from astroml.ingestion.horizon_stream import HorizonStreamingClient

# Threshold chosen generously above observed local timings so the test is a
# regression guard against real slowdowns (e.g. an accidentally-quadratic
# de-dupe window) rather than a flaky micro-benchmark tied to CI hardware.
_P95_LATENCY_BUDGET_SECONDS = 0.01

# Transactions processed per benchmark round. Large enough that a handful of
# GC pauses or scheduler jitter don't dominate the p95, small enough that the
# whole suite still runs in well under a second.
_TRANSACTIONS_PER_ROUND = 500


def _make_payload(paging_token: str) -> str:
    """A representative Horizon transaction payload, JSON-encoded."""
    return json.dumps(
        {
            "id": f"tx-{paging_token}",
            "paging_token": paging_token,
            "source_account": "GABCDEF...",
            "type": "payment",
            "amount": "100.0000000",
            "successful": True,
        }
    )


def _process_transactions(client: HorizonStreamingClient, count: int) -> None:
    """Feed ``count`` distinct transactions through the client's handling path.

    Runs synchronously (via ``asyncio.run``) inside the function pytest-benchmark
    times, so each benchmark round measures the full decode/de-dupe/dispatch
    cost for a burst of ``count`` transactions rather than a single call.
    """

    async def run() -> None:
        for i in range(count):
            await client._handle_payload(_make_payload(str(i)), lambda _tx: None)

    asyncio.run(run())


@pytest.mark.benchmark(group="fraud-ingestion-horizon-stream")
def test_horizon_stream_transaction_handling_p95_latency(benchmark):
    """p95 per-round latency for processing a burst of distinct transactions.

    Threshold: p95 round latency should stay under
    ``_P95_LATENCY_BUDGET_SECONDS`` for ``_TRANSACTIONS_PER_ROUND`` distinct
    (non-duplicate) transactions, so fraud scoring is never starved by
    ingestion-side latency regressions.
    """
    client = HorizonStreamingClient()

    benchmark.pedantic(
        _process_transactions,
        args=(client, _TRANSACTIONS_PER_ROUND),
        rounds=10,
        iterations=1,
    )

    timings = benchmark.stats.stats.data
    assert timings, "benchmark recorded no timing samples"

    p95 = statistics.quantiles(timings, n=100)[94] if len(timings) > 1 else timings[0]
    assert p95 < _P95_LATENCY_BUDGET_SECONDS, (
        f"p95 latency {p95:.4f}s for {_TRANSACTIONS_PER_ROUND} transactions "
        f"exceeds budget of {_P95_LATENCY_BUDGET_SECONDS}s"
    )


@pytest.mark.benchmark(group="fraud-ingestion-horizon-stream")
def test_horizon_stream_dedup_does_not_degrade_p95_under_replay_load(benchmark):
    """De-duplicated replay traffic must not blow the same p95 budget.

    Horizon's at-least-once delivery means a reconnect-heavy period can
    replay a large fraction of recently-seen transactions; the de-dupe path
    (``_already_delivered`` / ``_remember``) runs for every one of them and
    must stay cheap even when nearly everything is a duplicate.
    """
    client = HorizonStreamingClient(dedupe_capacity=1024)

    async def run(count: int) -> None:
        # First pass primes the de-dupe window; second pass is 100% replays,
        # the worst case for the lookup path this benchmark is guarding.
        for i in range(count):
            await client._handle_payload(_make_payload(str(i)), lambda _tx: None)
        for i in range(count):
            await client._handle_payload(_make_payload(str(i)), lambda _tx: None)

    def run_sync(count: int) -> None:
        asyncio.run(run(count))

    benchmark.pedantic(
        run_sync,
        args=(_TRANSACTIONS_PER_ROUND,),
        rounds=10,
        iterations=1,
    )

    timings = benchmark.stats.stats.data
    assert timings, "benchmark recorded no timing samples"

    p95 = statistics.quantiles(timings, n=100)[94] if len(timings) > 1 else timings[0]
    # Twice the transaction volume (fresh + replay pass), so double the budget.
    assert p95 < _P95_LATENCY_BUDGET_SECONDS * 2, (
        f"p95 latency {p95:.4f}s for {_TRANSACTIONS_PER_ROUND} transactions x2 "
        f"(fresh + replay) exceeds budget of {_P95_LATENCY_BUDGET_SECONDS * 2}s"
    )
