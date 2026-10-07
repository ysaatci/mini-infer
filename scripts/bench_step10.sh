#!/usr/bin/env bash
# Step 10 adaptive speculation: load sweep against fixed policies, then quiet-busy-quiet load over time.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
RUN="bash scripts/wsl_run.sh python"

$RUN -m bench.spec_run --out bench/results/step10-spec.json 2>/dev/null | grep -E "tok/s"
$RUN -m bench.adaptive_trace --out bench/results/step10-trace.json 2>/dev/null | grep -E "median|k chosen"
