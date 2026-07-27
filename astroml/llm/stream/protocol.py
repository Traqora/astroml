import json
from typing import Any, Dict

class StreamProtocol:
    @staticmethod
    def format_sse(event_type: str, data: Dict[str, Any]) -> str:
        payload = {
            "event": event_type,
            "data": data
        }
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def format_ws(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event": event_type,
            "data": data
        }
