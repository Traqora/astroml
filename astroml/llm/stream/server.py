import asyncio
import time
from typing import AsyncGenerator, Dict, Any
from .buffer import AdaptiveBuffer
from .reconnect import get_reconnection_manager

class StreamingServer:
    def __init__(self):
        pass

    async def stream_tokens(self, text: str, session_id: str, client_rtt_ms: float = 0.0) -> AsyncGenerator[str, None]:
        buffer = AdaptiveBuffer()
        buffer.adjust_backpressure(client_rtt_ms)
        reconnect_mgr = get_reconnection_manager()
        
        words = text.split(" ")
        for i, word in enumerate(words):
            # Append space
            token = word + (" " if i < len(words) - 1 else "")
            
            # Record in reconnect session history
            reconnect_mgr.append_tokens(session_id, [token])
            
            # Send through buffer
            buffered = buffer.add(token)
            for b_tok in buffered:
                yield b_tok
                
            # Inter-token latency: sleep roughly 50ms per token
            await asyncio.sleep(0.05)
            
        # Final flush
        leftovers = buffer.flush()
        for b_tok in leftovers:
            yield b_tok

_server = StreamingServer()

def get_streaming_server() -> StreamingServer:
    return _server
