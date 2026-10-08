"""Speculation under load that changes over time: quiet, busy, quiet again.

Chat requests arrive as a Poisson process whose rate steps up and back down. Each policy sees the same
arrival times. For the adaptive policy the run also records, per decode step, the batch size and the k it
chose, to show it speculating while quiet and backing off while busy.

python -m bench.adaptive_trace --out bench/results/step10-trace.json
"""

import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig

from bench.metrics import summarize_batch
from bench.report import markdown_table, save_json
from bench.spec_run import PROMPTS, make_policy
from mini_infer.draft_policy import AdaptiveDraftPolicy
from mini_infer.engine import LLMEngine
from mini_infer.loader import load_model, resolve_model_dir
from mini_infer.paged_cache import DEFAULT_BLOCK_SIZE, PagedKVPool
from mini_infer.speculative import SpeculativeConfig

PHASES = [(40.0, 0.3), (30.0, 4.0), (40.0, 0.3)]  # (seconds, requests per second)


def arrivals(seed: int = 0) -> list[float]:
    rng, times, clock = random.Random(seed), [], 0.0
    phase_start = 0.0
    for duration, rate in PHASES:
        while True:
            clock += rng.expovariate(rate)
            if clock >= phase_start + duration:
                clock = phase_start + duration
                break
            times.append(clock)
        phase_start += duration
    return times


def drive(engine: LLMEngine, prompts: list[list[int]], arrival_s: list[float], stop_ids):
    """Returns token arrival times per request, and per decode step (seconds, batch size, k)."""
    pending = list(enumerate(arrival_s))
    token_times: dict[str, list[float]] = defaultdict(list)
    steps: list[tuple[float, int, int]] = []
    start = time.perf_counter()
    while pending or engine.has_unfinished():
        now = time.perf_counter() - start
        while pending and pending[0][1] <= now:
            i, _ = pending.pop(0)
            engine.add_request(prompts[i % len(prompts)], 256, stop_ids=stop_ids, request_id=str(i))
        if not engine.has_unfinished():
            time.sleep(max(0.0, pending[0][1] - now))
            continue
        for out in engine.step():
            token_times[out.request_id].append(time.perf_counter() - start)
        batch = engine.last_batch
        if batch.decode:
            steps.append((time.perf_counter() - start, len(batch.decode), batch.lookahead - 1))
    return token_times, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--policies", nargs="+", default=["none", "fixed-2", "adaptive"])
    parser.add_argument("--kv-cache-gb", type=float, default=1.41)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    eos = GenerationConfig.from_pretrained(model_dir).eos_token_id
    stop_ids = frozenset([eos] if isinstance(eos, int) else eos)
    prompts = [
        tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False),
                  add_special_tokens=False).input_ids
        for p in PROMPTS
    ]
    target, draft = load_model(args.model), load_model(args.draft_model)
    num_blocks = PagedKVPool.blocks_for_memory(target.config, int(args.kv_cache_gb * 1e9), DEFAULT_BLOCK_SIZE, torch.bfloat16)
    arrival_s = arrivals()

    # One engine with the policy swapped between runs, so every policy sees the same compiled kernels and
    # memory state, and runs follow each other closely (separate engines measured clock drift too).
    policies = {name: make_policy(name) for name in args.policies}
    adaptive = policies.get("adaptive", AdaptiveDraftPolicy())  # the engine calibrates it at warmup
    engine = LLMEngine(target, num_blocks, max_batch_size=64, speculative=SpeculativeConfig(draft, adaptive, max_batch_size=16))
    engine.warmup()
    arrivals_by_id = {str(i): t for i, t in enumerate(arrival_s)}

    rows, trace = [], []
    for name, policy in policies.items():
        engine.speculative.policy = policy
        token_times, steps = drive(engine, prompts, arrival_s, stop_ids)
        metrics = summarize_batch(token_times, arrivals_by_id, max(t[-1] for t in token_times.values()), 0)
        rows.append((name, "quiet-busy-quiet", metrics))
        print(f"{name:10s} median request {metrics.e2e_p50_s:.2f} s, p99 {metrics.e2e_p99_s:.2f} s", flush=True)
        if name == "adaptive":
            trace = steps  # every decode step: above 16 requests k is 0 by rule (plain decode)
            print(f"  k chosen: {dict(sorted(Counter(k for *_, k in trace).items()))}")

    print("\n" + markdown_table(rows))
    save_json(args.out, vars(args) | {"out": str(args.out), "phases": PHASES, "requests": len(arrival_s)}, rows,
              {"adaptive_trace": trace})


if __name__ == "__main__":
    main()
