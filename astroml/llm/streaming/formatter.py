"""Stream formatters for Server-Sent Events (SSE) and WebSockets."""
from __future__ import annotations

import json
from typing import Optional, Dict, Any


def format_sse(token: Optional[str], finished: bool = False, usage: Optional[Dict[str, int]] = None) -> str:
    """Format token output into SSE data packet format."""
    payload: Dict[str, Any] = {
        "token": token,
        "finished": finished
    }
    if usage:
        payload["usage"] = usage
        
    # SSE lines must start with 'data: ' and end with '\n\n'
    return f"data: {json.dumps(payload)}\n\n"


def format_websocket(token: Optional[str], finished: bool = False, usage: Optional[Dict[str, int]] = None) -> str:
    """Format token output into WebSocket payload string."""
    if finished:
        payload = {
            "type": "done",
            "usage": usage or {"total_tokens": 0}
        }
    else:
        payload = {
            "type": "token",
            "content": token or ""
        }
    return json.dumps(payload)
