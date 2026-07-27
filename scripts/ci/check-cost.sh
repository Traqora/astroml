#!/bin/bash
set -euo pipefail

COST_BASELINE_FILE=".llm_cost_baseline.json"
COST_THRESHOLD_PCT="${COST_THRESHOLD_PCT:-20}"

echo "Checking LLM cost against baseline..."

CURRENT_COST="0.0"
if [ -f "$COST_BASELINE_FILE" ]; then
    BASELINE_COST=$(python3 -c "import json; print(json.load(open('$COST_BASELINE_FILE')).get('total_cost', 0))")
    PCT_DIFF=$(python3 -c "
baseline = float('$BASELINE_COST')
current = float('$CURRENT_COST')
if baseline > 0:
    diff = ((current - baseline) / baseline) * 100
    print(f'{diff:.1f}')
else:
    print('0')
")
    echo "Baseline cost: \$${BASELINE_COST}"
    echo "Current cost:  \$${CURRENT_COST}"
    echo "Change:        ${PCT_DIFF}%"

    THRESHOLD="$COST_THRESHOLD_PCT"
    if python3 -c "import sys; sys.exit(0 if float('$PCT_DIFF') > float('$THRESHOLD') else 1)"; then
        echo "WARNING: Cost increased by ${PCT_DIFF}% (threshold: ${THRESHOLD}%)"
        echo "Review LLM API usage for unexpected increases."
    else
        echo "Cost within acceptable range."
    fi
else
    echo "No cost baseline found. Creating baseline..."
    echo "{\"total_cost\": 0.0, \"created_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$COST_BASELINE_FILE"
    echo "Baseline created at \$${CURRENT_COST}"
fi
