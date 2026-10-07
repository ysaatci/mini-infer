#!/usr/bin/env bash
# Step 5 paged KV cache: same KV memory as step 4 (1.41 GB), batch cap 64, mini-infer vs vLLM.
# Plus the single-request benchmark, to check paging didn't slow down one request.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
MINI="bash scripts/wsl_run.sh python"
VLLM="/root/venvs/vllm/bin/python"
OUT=bench/results/step5

$MINI -m bench.run --engines mini-infer --out $OUT-single.json 2>/dev/null | tail -5

for engine in mini-infer vllm; do
  py=$MINI; [ "$engine" = vllm ] && py=$VLLM
  $py -m bench.batch_run --engine $engine --num-requests 200 --out $OUT-$engine-offline.json 2>/dev/null | tail -2
  for rate in 1 2 3; do
    $py -m bench.batch_run --engine $engine --num-requests 100 --rate $rate --out $OUT-$engine-rate$rate.json 2>/dev/null | tail -2
  done
done
