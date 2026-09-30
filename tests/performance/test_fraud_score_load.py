"""P95 load test for POST /api/v1/fraud/score (issue #952).

Benchmarks the real-time scoring endpoint under a realistic-shaped
request (near the max 50 accounts / 500 edges the schema allows) and
asserts the p95 latency stays under threshold, so a regression in the
request-handling path (not model inference itself, which is mocked out
via the scorer-unavailable fallback — see Scope below) gets caught in CI.

Threshold: p95 < 200ms for a max-size request against the FastAPI
TestClient (in-process, no network hop) with no scorer model loaded.

Scope: `load_scorer()` returns None when no model checkpoint is
configured (the default in CI/test environments), so `score_accounts`
takes its all-zero fallback path rather than running real model
inference. This deliberately isolates the endpoint's own overhead
(request validation, response serialization, FastAPI/Starlette
routing) from model inference latency, which depends on whatever model
is loaded and belongs in a model-level benchmark instead
(tests/performance/test_benchmarks.py's `test_model_inference_batch_100`
covers that separately). A future load test against a loaded model
would need its own threshold, since inference latency is not comparable
to routing/validation overhead.
"""

from __future__ import annotations

import pytest

_ACCOUNT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"  # matches _STELLAR_ACCOUNT_RE: [A-Z2-7]


def _make_public_key(seed: int) -> str:
    """Deterministic 56-char fake Stellar account id (G + 55 chars from
    the base32-style alphabet _STELLAR_ACCOUNT_RE actually requires).

    ScoreRequest only validates the `G` prefix, length, and character set
    via that regex, not a real ed25519 checksum, so this is sufficient
    without a real keypair - but it must still match the regex or every
    request in this test would 422 instead of exercising the 200 path.
    """
    chars = []
    n = seed
    for _ in range(55):
        chars.append(_ACCOUNT_ALPHABET[n % len(_ACCOUNT_ALPHABET)])
        n //= len(_ACCOUNT_ALPHABET)
    return "G" + "".join(chars)


def _make_score_request(num_accounts: int, num_edges: int) -> dict:
    accounts = [_make_public_key(i) for i in range(num_accounts)]
    edges = [
        {
            "src": accounts[i % num_accounts],
            "dst": accounts[(i + 1) % num_accounts],
            "amount": 100.5,
            "timestamp": 1_700_000_000.0 + i,
            "asset": "XLM",
        }
        for i in range(num_edges)
    ]
    return {"accounts": accounts, "edges": edges}


@pytest.mark.xdist_group("api_fraud")
@pytest.mark.benchmark(group="fraud-score-p95")
def test_fraud_score_p95_latency(benchmark, client):
    """Benchmark POST /fraud/score at max request size; assert p95 < 200ms."""
    payload = _make_score_request(num_accounts=50, num_edges=500)

    def _score() -> int:
        resp = client.post("/api/v1/fraud/score", json=payload)
        assert resp.status_code == 200
        return resp.status_code

    benchmark.pedantic(_score, rounds=30, iterations=1, warmup_rounds=5)

    # pytest-benchmark's Stats has no built-in percentile method; compute
    # p95 directly from the sorted raw samples.
    samples = benchmark.stats.stats.sorted_data
    p95_index = int(len(samples) * 0.95)
    p95 = samples[min(p95_index, len(samples) - 1)]
    assert p95 < 0.2, f"p95 latency {p95:.3f}s exceeds 200ms threshold"


@pytest.mark.xdist_group("api_fraud")
def test_fraud_score_response_shape_for_max_request(client):
    """A max-size request returns one score per requested account."""
    payload = _make_score_request(num_accounts=50, num_edges=500)

    resp = client.post("/api/v1/fraud/score", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert set(body["scores"].keys()) == set(payload["accounts"])
