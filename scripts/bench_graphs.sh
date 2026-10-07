#!/usr/bin/env bash
# CUDA graphs: single request (with/without graphs, HF) and batching (offline + online) vs vLLM's step 5 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
MINI="bash scripts/wsl_run.sh python"
OUT=bench/results/graphs

$MINI -m bench.run --engines mini-infer mini-infer-eager hf --out $OUT-single.json 2>/dev/null | tail -12
$MINI -m bench.batch_run --engine mini-infer --num-requests 200 --out $OUT-mini-infer-offline.json 2>/dev/null | tail -2
for rate in 1 2 3; do
  $MINI -m bench.batch_run --engine mini-infer --num-requests 100 --rate $rate --out $OUT-mini-infer-rate$rate.json 2>/dev/null | tail -2
done
