#!/bin/bash
set -euo pipefail

BASELINE_FILE=".llm_eval_baseline.json"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-5.0}"

echo "Running LLM evaluation test suite..."
# NOTE: previously this line ended in `|| true`, which swallowed every
# pytest failure unconditionally — meaning this script could never fail
# here no matter what broke in tests/llm/eval/. That silenced exactly the
# regressions this CI job exists to catch, so it has been removed: a real
# test failure in the eval suite now fails the script immediately, same as
# every other CI step in this repo.
python3 -m pytest tests/llm/eval/ -v --tb=short 2>&1 | tee .llm_eval_output.txt

echo ""
echo "Running LLM evaluation benchmarks and enforcing quality gate..."
# This is the piece that was missing entirely: an actual numeric score is
# computed (see astroml/llm/eval/threshold.py) and compared against
# SCORE_THRESHOLD. The previous version of this script only ever *printed*
# the threshold value — it never ran a comparison and never exited
# non-zero, so the "quality gate" always passed regardless of the real
# eval score.
SAVE_BASELINE_FLAG=""
if [ ! -f "$BASELINE_FILE" ]; then
    SAVE_BASELINE_FLAG="--save-baseline"
fi

python3 scripts/ci/run_eval_check.py --threshold "$SCORE_THRESHOLD" $SAVE_BASELINE_FLAG
