#!/usr/bin/env bash
# Step 11: what deterministic mode costs. Normal and deterministic runs alternate, so clock drift
# on this laptop GPU affects both alike.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
RUN="bash scripts/wsl_run.sh python"
OUT=bench/results/step11

$RUN -m bench.run --engines mini-infer mini-infer-det --out $OUT-single.json 2>/dev/null | tail -8
for mode in "" "--deterministic" "" "--deterministic"; do
  $RUN -m bench.batch_run --engine mini-infer $mode --num-requests 200 --out $OUT-offline${mode:+-det}.json 2>/dev/null | tail -2 | head -1
done
for rate in 1 2 3; do
  for mode in "" "--deterministic"; do
    $RUN -m bench.batch_run --engine mini-infer $mode --num-requests 100 --rate $rate --out $OUT-rate$rate${mode:+-det}.json 2>/dev/null | tail -2 | head -1
  done
done
