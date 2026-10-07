#!/usr/bin/env bash
# Step 6: the same online/offline workloads as bench_graphs.sh, sent over HTTP to the server.
# Compare with bench/results/graphs-mini-infer-*.json (same engine, called directly).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
RUN="bash scripts/wsl_run.sh python"
OUT=bench/results/step6-http

$RUN -m mini_infer.server --port 8000 > /tmp/mini-infer-server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
until curl -s localhost:8000/v1/models > /dev/null; do sleep 1; done

$RUN -m bench.http_run --num-requests 200 --out $OUT-offline.json 2>/dev/null | tail -1
for rate in 1 2 3; do
  $RUN -m bench.http_run --num-requests 100 --rate $rate --out $OUT-rate$rate.json 2>/dev/null | tail -1
done
