# Structured logging across pipeline components
import json
def log_structured(event, data):
    print(json.dumps({"event": event, "data": data}))

LOGGING_STAGES = ("ingestion", "features", "training")

def with_stage(data, stage):
    return {**data, "stage": stage}
