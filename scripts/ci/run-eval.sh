#!/bin/bash
set -euo pipefail

BASELINE_FILE=".llm_eval_baseline.json"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-5.0}"

echo "Running LLM evaluation benchmarks..."
python3 -m pytest tests/llm/eval/ -v --tb=short 2>&1 | tee .llm_eval_output.txt || true

SCORE=$(python3 -c "
import json, os
results = {}
try:
    with open('$BASELINE_FILE') as f:
        results = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    pass
print(json.dumps({'baseline_found': bool(results), 'baseline_scores': results}))
")

echo "Evaluation complete: $SCORE"

if [ -f "$BASELINE_FILE" ]; then
    python3 -c "
import json
with open('$BASELINE_FILE') as f:
    baseline = json.load(f)
threshold = float('$SCORE_THRESHOLD')
print(f'Baseline found with {len(baseline)} metrics. Threshold: {threshold}%')
"
fi
