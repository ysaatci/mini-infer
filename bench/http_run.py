"""Online benchmark over HTTP against a running OpenAI-compatible server.

Same requests as bench.batch_run (same seed), sent as streaming /v1/completions calls, so the
difference to the direct-engine numbers is what serving costs: HTTP, JSON, SSE, detokenization,
and the hop between the event loop and the engine thread.

python -m mini_infer.server &
python -m bench.http_run --rate 2 --out bench/results/http-rate2.json
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from bench.batch_workload import BatchRequest, make_requests
from bench.metrics import summarize_batch
from bench.report import markdown_table, save_json


async def send(client: httpx.AsyncClient, model: str, request: BatchRequest, start: float, token_times: dict) -> None:
    await asyncio.sleep(max(0.0, request.arrival_s - (time.perf_counter() - start)))
    body = {
        "model": model,
        "prompt": request.prompt_ids,
        "max_tokens": request.output_len,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,  # fixed output length, as in the engine benchmark
    }
    async with client.stream("POST", "/v1/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            # One chunk per token here: the prompts are plain English, so no token's text is held back.
            if json.loads(line[6:])["choices"][0]["text"]:
                token_times[request.id].append(time.perf_counter() - start)


async def run(url: str, model: str, requests: list[BatchRequest]) -> tuple[dict[str, list[float]], float]:
    token_times: dict[str, list[float]] = defaultdict(list)
    limits = httpx.Limits(max_connections=None)  # every request gets its own connection, like separate users
    async with httpx.AsyncClient(base_url=url, timeout=None, limits=limits) as client:
        start = time.perf_counter()
        await asyncio.gather(*(send(client, model, r, start, token_times) for r in requests))
        return token_times, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--rate", type=float, default=None, help="requests/s; omit for offline (all at once)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    warmup = make_requests(tokenizer, 4, None, seed=999, output_range=(8, 8))
    asyncio.run(run(args.url, args.model, warmup))

    requests = make_requests(tokenizer, args.num_requests, args.rate, args.seed)
    token_times, makespan = asyncio.run(run(args.url, args.model, requests))
    # Peak GPU memory isn't visible from the client side, so it's reported as 0.
    metrics = summarize_batch(token_times, {r.id: r.arrival_s for r in requests}, makespan, peak_mem_bytes=0)

    workload = "offline" if args.rate is None else f"rate{args.rate:g}"
    rows = [("mini-infer-http", f"{workload}-n{args.num_requests}", metrics)]
    print(markdown_table(rows))
    save_json(args.out, vars(args) | {"out": str(args.out)}, rows)


if __name__ == "__main__":
    main()
