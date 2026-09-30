# Structured logging across pipeline components
import json
def log_structured(event, data):
    print(json.dumps({"event": event, "data": data}))
