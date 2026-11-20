# Structured logging across pipeline components
import json, time, uuid
RUN_ID = uuid.uuid4().hex[:12]
def log_structured(event, data):
    print(json.dumps({"ts": time.time(), "run_id": RUN_ID, "stage": data.pop("stage", "v1"), "event": event, "data": data}))
