"""LLM Streaming response processing package."""
from __future__ import annotations

from astroml.llm.streaming.buffer import StreamBuffer
from astroml.llm.streaming.handler import StreamHandler
from astroml.llm.streaming.formatter import format_sse, format_websocket
from astroml.llm.streaming.aggregator import StreamAggregator

__all__ = [
    "StreamBuffer",
    "StreamHandler",
    "format_sse",
    "format_websocket",
    "StreamAggregator",
]
