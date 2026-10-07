#!/usr/bin/env bash
# Step 4 batching benchmark: offline throughput and online load, mini-infer vs vLLM.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
MINI="bash scripts/wsl_run.sh python"
VLLM="/root/venvs/vllm/bin/python"
OUT=bench/results/step4

for engine in mini-infer vllm; do
  py=$MINI; [ "$engine" = vllm ] && py=$VLLM
  $py -m bench.batch_run --engine $engine --num-requests 200 --out $OUT-$engine-offline.json 2>/dev/null | tail -1
  for rate in 0.5 1 2; do
    $py -m bench.batch_run --engine $engine --num-requests 100 --rate $rate --out $OUT-$engine-rate$rate.json 2>/dev/null | tail -1
  done
done
