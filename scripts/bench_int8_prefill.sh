#!/usr/bin/env bash
# int8 after moving prefill onto the Triton kernel: quality (prefill path changed), single request, batching.
# Compare with bench/results/step8-*.json (before) and graphs-*.json (bf16).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
RUN="bash scripts/wsl_run.sh python"
OUT=bench/results/step8b

$RUN -m bench.quality --out $OUT-quality.json 2>/dev/null | tail -3
$RUN -m bench.run --engines mini-infer mini-infer-int8 --out $OUT-single.json 2>/dev/null | tail -8
$RUN -m bench.batch_run --engine mini-infer --int8 --num-requests 200 --out $OUT-int8-offline.json 2>/dev/null | tail -2
for rate in 1 2 3; do
  $RUN -m bench.batch_run --engine mini-infer --int8 --num-requests 100 --rate $rate --out $OUT-int8-rate$rate.json 2>/dev/null | tail -2
done
