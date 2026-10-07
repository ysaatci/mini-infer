"""Batching benchmark: many mixed-length requests, offline (all at once) or online (Poisson arrivals).

python -m bench.batch_run --engine mini-infer --out bench/results/step4-mini-offline.json
python -m bench.batch_run --engine mini-infer --rate 2 --out bench/results/step4-mini-rate2.json
vLLM runs from its own venv: /root/venvs/vllm/bin/python -m bench.batch_run --engine vllm ...
"""

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bench.batch_engines import BATCH_ENGINES, BatchEngine
from bench.batch_workload import BatchRequest, make_requests
from bench.metrics import summarize_batch
from bench.report import markdown_table, save_json


def drive(engine: BatchEngine, requests: list[BatchRequest]) -> tuple[dict[str, list[float]], float]:
    """Feed requests in at their arrival times and step the engine. Returns token times and makespan."""
    pending = deque(sorted(requests, key=lambda r: r.arrival_s))
    token_times: dict[str, list[float]] = defaultdict(list)
    start = time.perf_counter()
    while pending or engine.has_unfinished():
        now = time.perf_counter() - start
        while pending and pending[0].arrival_s <= now:
            engine.add(pending.popleft())
        if not engine.has_unfinished():
            time.sleep(pending[0].arrival_s - now)  # idle until the next user shows up
            continue
        outputs = engine.step()
        t = time.perf_counter() - start
        for request_id, new_tokens, _ in outputs:
            token_times[request_id].extend([t] * new_tokens)
    return token_times, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=list(BATCH_ENGINES))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--rate", type=float, default=None, help="requests/s; omit for offline (all at once)")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=1536, help="longest prompt + output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    requests = make_requests(tokenizer, args.num_requests, args.rate, args.seed)
    engine = BATCH_ENGINES[args.engine](args.model, args.max_batch_size, args.max_len)

    warmup = make_requests(tokenizer, 4, None, seed=999, output_range=(8, 8))
    drive(engine, [BatchRequest(f"warmup-{r.id}", r.prompt_ids, r.output_len, 0.0) for r in warmup])

    torch.cuda.reset_peak_memory_stats()
    token_times, makespan = drive(engine, requests)
    metrics = summarize_batch(token_times, {r.id: r.arrival_s for r in requests}, makespan, torch.cuda.max_memory_allocated())

    workload = "offline" if args.rate is None else f"rate{args.rate:g}"
    rows = [(args.engine, f"{workload}-n{args.num_requests}", metrics)]
    print(markdown_table(rows))
    save_json(args.out, vars(args) | {"out": str(args.out)}, rows)


if __name__ == "__main__":
    main()
