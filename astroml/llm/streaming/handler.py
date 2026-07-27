"""LLM Stream Processing Handler."""
from __future__ import annotations

import asyncio
import time
import logging
from typing import AsyncIterator, Callable, Optional, Dict, Any, List

from astroml.llm.streaming.buffer import StreamBuffer

logger = logging.getLogger(__name__)


class StreamHandler:
    """Manages an active LLM streaming generation session."""

    def __init__(self, session_id: str, buffer_max_size: int = 100):
        self.session_id = session_id
        self.buffer = StreamBuffer[str](max_size=buffer_max_size)
        self.start_time: float = 0.0
        self.first_token_time: Optional[float] = None
        self.token_count = 0
        self.is_running = False

    async def process_stream(
        self,
        generator: AsyncIterator[str],
        on_token_callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncIterator[str]:
        """
        Process the raw generator, feed the stream buffer, and yield tokens.
        Measures first-token latency, tokens/sec, and supports cancellation.
        """
        self.start_time = time.perf_counter()
        self.is_running = True
        self.token_count = 0
        
        try:
            async for chunk in generator:
                if self.buffer.is_aborted:
                    logger.info("Stream handler for session %s aborted", self.session_id)
                    break
                    
                if self.first_token_time is None:
                    self.first_token_time = time.perf_counter()
                    latency_ms = (self.first_token_time - self.start_time) * 1000
                    logger.info("First token latency: %.2fms", latency_ms)
                    
                self.token_count += 1
                
                # Push to buffer
                success = await self.buffer.push(chunk)
                if not success:
                    break
                    
                if on_token_callback:
                    on_token_callback(chunk)
                    
                yield chunk
                
        except asyncio.CancelledError:
            logger.info("Stream execution cancelled for session %s", self.session_id)
            self.buffer.abort()
            raise
        except Exception as e:
            logger.error("Exception during streaming for session %s: %s", self.session_id, e)
            self.buffer.abort()
            raise
        finally:
            self.is_running = False

    def abort(self) -> None:
        """Cancel/abort the active stream."""
        self.buffer.abort()
        self.is_running = False

    def get_metadata(self, finish_reason: str = "stop") -> Dict[str, Any]:
        """Calculate and return stream speed, count, and status metadata."""
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        tokens_per_sec = self.token_count / duration if duration > 0 else 0.0
        
        first_token_latency_ms = (
            (self.first_token_time - self.start_time) * 1000
            if self.first_token_time
            else 0.0
        )
        
        return {
            "session_id": self.session_id,
            "total_tokens": self.token_count,
            "duration_seconds": duration,
            "tokens_per_second": tokens_per_sec,
            "first_token_latency_ms": first_token_latency_ms,
            "finish_reason": finish_reason,
        }
