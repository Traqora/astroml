"""Regression tests for issues #825, #826, #827, #828."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def load_module(name: str, path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


metrics_mod = load_module("astroml.benchmarking.metrics", "astroml/benchmarking/metrics.py")
redis_mod = load_module("astroml.cache.redis_cache", "astroml/cache/redis_cache.py")
semantic_mod = load_module("astroml.cache.llm_semantic_cache", "astroml/cache/llm_semantic_cache.py")
stream_mod = load_module("api.routers.stream", "api/routers/stream.py")

LinkPredictionMetrics = metrics_mod.LinkPredictionMetrics
RedisCache = redis_mod.RedisCache
LLMEmbeddingProvider = semantic_mod.LLMEmbeddingProvider
DefaultNoopEmbeddingProvider = semantic_mod.DefaultNoopEmbeddingProvider
websocket_stream = stream_mod.ws_stream


# --- Issue #825: metrics bare except ---
def test_metrics_ranking_exception_handling(caplog):
    """Test that ranking metrics computation catches exceptions and logs warnings rather than using bare except."""
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])

    with patch.object(LinkPredictionMetrics, "_hits_at_k", side_effect=ZeroDivisionError("Division by zero")):
        with caplog.at_level(logging.WARNING):
            metrics = LinkPredictionMetrics.compute(y_true, y_prob > 0.5, y_prob, k_values=[2])

    assert metrics["hits_at_2"] == 0.0
    assert metrics["precision_at_2"] == 0.0
    assert metrics["recall_at_2"] == 0.0
    assert "Failed computing ranking metrics at k=2" in caplog.text


# --- Issue #826: Replace bare excepts in redis_cache ---
def test_redis_cache_dataframe_hash_and_json_decode(caplog):
    """Test that RedisCache._hash_key and get handle exceptions cleanly with debug logs."""
    cache = RedisCache.__new__(RedisCache)
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Simulate DataFrame head(100).to_csv throwing an error
    with patch.object(df, "head", side_effect=ValueError("Sample error")):
        with caplog.at_level(logging.DEBUG):
            key = cache._hash_key(df, extra=df)
            assert isinstance(key, str)
            assert len(key) == 16
            assert "Failed to hash DataFrame sample" in caplog.text

    # Test JSON decode failure in get()
    cache._client = MagicMock()
    cache._client.get.return_value = b"invalid json bytes that cannot unpickle"
    cache._stats = MagicMock()

    with caplog.at_level(logging.DEBUG):
        result = cache.get("test-key")
        assert result == b"invalid json bytes that cannot unpickle"
        assert "Failed to decode JSON from cached data" in caplog.text


# --- Issue #827: Replace bare except in WS stream path ---
@pytest.mark.asyncio
async def test_websocket_stream_send_error_handling(caplog):
    """Test that WebSocket error handling catches disconnect/send exceptions and logs warnings."""
    mock_ws = AsyncMock()
    mock_ws.receive_text.side_effect = Exception("Simulated stream processing failure")
    # send_json also fails when trying to send error to client
    mock_ws.send_json.side_effect = RuntimeError("Socket connection closed prematurely")

    with caplog.at_level(logging.WARNING):
        await websocket_stream(mock_ws)

    assert "Error during WebSocket streaming" in caplog.text
    assert "Failed to send WebSocket error message to client" in caplog.text
    mock_ws.close.assert_awaited_once()


# --- Issue #828: Implement semantic-cache embed ---
def test_llm_embedding_provider_default_embed():
    """Test that LLMEmbeddingProvider and DefaultNoopEmbeddingProvider provide working embed() methods."""
    base_provider = LLMEmbeddingProvider()
    vec1 = base_provider.embed(text="Detect fraud pattern in Stellar transactions", model="default")
    assert isinstance(vec1, list)
    assert len(vec1) == 64
    # Check that vector is normalized (magnitude approx 1.0)
    norm = sum(x * x for x in vec1) ** 0.5
    assert abs(norm - 1.0) < 1e-4

    # Check determinism
    vec2 = base_provider.embed(text="Detect fraud pattern in Stellar transactions", model="default")
    assert vec1 == vec2

    # Check different text produces different vector
    vec3 = base_provider.embed(text="Different query content entirely", model="default")
    assert vec1 != vec3

    # DefaultNoopEmbeddingProvider works without throwing RuntimeError or NotImplementedError
    noop_provider = DefaultNoopEmbeddingProvider()
    vec_noop = noop_provider.embed(text="Detect fraud pattern in Stellar transactions", model="default")
    assert vec_noop == vec1


if __name__ == "__main__":
    caplog = pytest.LogCaptureFixture(None)
    test_metrics_ranking_exception_handling(caplog)
    print("test_metrics_ranking_exception_handling: PASSED")

    caplog = pytest.LogCaptureFixture(None)
    test_redis_cache_dataframe_hash_and_json_decode(caplog)
    print("test_redis_cache_dataframe_hash_and_json_decode: PASSED")

    asyncio.run(test_websocket_stream_send_error_handling(caplog))
    print("test_websocket_stream_send_error_handling: PASSED")

    test_llm_embedding_provider_default_embed()
    print("test_llm_embedding_provider_default_embed: PASSED")
    print("\nALL 4 TESTS PASSED!")
