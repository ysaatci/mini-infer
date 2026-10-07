#!/usr/bin/env bash
# Step 8 int8 weights: single request, batching (same workloads as bench_graphs.sh), and with speculation.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
RUN="bash scripts/wsl_run.sh python"
OUT=bench/results/step8

$RUN -m bench.run --engines mini-infer mini-infer-int8 --out $OUT-single.json 2>/dev/null | tail -8
$RUN -m bench.batch_run --engine mini-infer --int8 --num-requests 200 --out $OUT-int8-offline.json 2>/dev/null | tail -2
for rate in 1 2 3; do
  $RUN -m bench.batch_run --engine mini-infer --int8 --num-requests 100 --rate $rate --out $OUT-int8-rate$rate.json 2>/dev/null | tail -2
done
$RUN -m bench.spec_run --int8 --draft-tokens 0 2 --concurrency 1 4 --temperatures 0 --out $OUT-int8-spec.json 2>/dev/null | tail -6
