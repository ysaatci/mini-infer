"""Speculative decoding benchmark on real chat prompts.

The other benchmarks repeat one passage, which a draft model predicts unrealistically well. Here every
request is a different chat question answered until the model's own end of turn (max 256 tokens).
With temperature 0 the output is identical with and without speculation, so speeds compare exactly.

python -m bench.spec_run --out bench/results/step7-spec.json
"""

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig

from bench.metrics import percentile
from bench.report import markdown_table, save_json
from mini_infer.engine import LLMEngine
from mini_infer.loader import load_model, resolve_model_dir
from mini_infer.paged_cache import DEFAULT_BLOCK_SIZE, PagedKVPool
from mini_infer.quant import quantize_model
from mini_infer.sampling import SamplingParams
from mini_infer.speculative import SpeculativeConfig

PROMPTS = json.loads((Path(__file__).parent / "chat_prompts.json").read_text())


@dataclass(frozen=True)
class SpecMetrics:
    output_tok_s: float  # all generated tokens / wall time
    decode_tok_s_p50: float  # per request, after its first token
    ttft_p50_ms: float
    acceptance: float  # share of drafted tokens accepted
    tokens_per_step: float  # tokens a request gains per target forward: 1 + acceptance * k


def run(engine: LLMEngine, prompts: list[list[int]], concurrency: int, params: SamplingParams, stop_ids) -> SpecMetrics:
    """Closed loop: keep `concurrency` requests in flight, start the next prompt when one finishes."""
    pending = list(enumerate(prompts))
    start_times, token_times = {}, {}
    if engine.speculative:
        engine.speculative.num_drafted = engine.speculative.num_accepted = 0
    start = time.perf_counter()
    in_flight = 0
    while pending or engine.has_unfinished():
        while pending and in_flight < concurrency:
            i, ids = pending.pop(0)
            engine.add_request(ids, 256, params, stop_ids, request_id=str(i))
            start_times[str(i)], token_times[str(i)] = time.perf_counter(), []
            in_flight += 1
        for out in engine.step():
            token_times[out.request_id].append(time.perf_counter())
            in_flight -= out.finished
    wall = time.perf_counter() - start

    decode = [(len(t) - 1) / (t[-1] - t[0]) for t in token_times.values() if len(t) > 1]
    ttft = [token_times[r][0] - start_times[r] for r in token_times]
    acceptance = engine.speculative.acceptance_rate if engine.speculative else 0.0
    k = engine.speculative.k if engine.speculative else 0
    return SpecMetrics(
        output_tok_s=sum(len(t) for t in token_times.values()) / wall,
        decode_tok_s_p50=statistics.median(decode),
        ttft_p50_ms=percentile(ttft, 50) * 1e3,
        acceptance=acceptance,
        tokens_per_step=1 + acceptance * k,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--draft-tokens", nargs="+", type=int, default=[0, 2, 4, 6], help="k; 0 = no speculation")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7])
    parser.add_argument("--kv-cache-gb", type=float, default=1.41)
    parser.add_argument("--int8", action="store_true", help="int8 weights for target and draft")
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
    load = (lambda name: quantize_model(load_model(name))) if args.int8 else load_model
    target, draft = load(args.model), load(args.draft_model)
    num_blocks = PagedKVPool.blocks_for_memory(target.config, int(args.kv_cache_gb * 1e9), DEFAULT_BLOCK_SIZE, torch.bfloat16)

    rows = []
    for k in args.draft_tokens:
        speculative = SpeculativeConfig(draft, k, max_batch_size=max(args.concurrency)) if k else None
        engine = LLMEngine(target, num_blocks, max_batch_size=max(args.concurrency), speculative=speculative)
        engine.warmup()
        for temperature in args.temperatures:
            for concurrency in args.concurrency:
                metrics = run(engine, prompts, concurrency, SamplingParams(temperature=temperature), stop_ids)
                label = (f"k={k}" if k else "no-spec") + ("-int8" if args.int8 else "")
                rows.append((label, f"c{concurrency}-t{temperature:g}", metrics))
                print(f"{rows[-1][0]:8s} {rows[-1][1]:10s} {metrics.output_tok_s:7.1f} tok/s  acceptance {metrics.acceptance:.2f}", flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + markdown_table(rows))
    save_json(args.out, vars(args) | {"out": str(args.out)}, rows)


if __name__ == "__main__":
    main()
