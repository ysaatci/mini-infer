"""Single-request latency benchmark.

python -m bench.run --engines mini-infer hf --prompt-lens 128 512 2048 --output-len 256
"""

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bench.engines import ENGINES, Engine
from bench.metrics import summarize
from bench.report import Row, markdown_table, save_json
from bench.workload import Workload, make_prompt
from mini_infer.loader import resolve_model_dir


def measure(engine: Engine, prompt_ids, workload: Workload, repeats: int):
    torch.cuda.reset_peak_memory_stats()
    runs = [engine.run(prompt_ids, workload.output_len) for _ in range(repeats)]
    return summarize(runs, torch.cuda.max_memory_allocated())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--engines", nargs="+", default=list(ENGINES), choices=list(ENGINES))
    parser.add_argument("--prompt-lens", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True, help="JSON results file, e.g. bench/results/step3.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_dir(args.model))
    workloads = [Workload(n, args.output_len) for n in args.prompt_lens]

    rows: list[Row] = []
    for name in args.engines:
        engine = ENGINES[name](args.model)
        engine.run(make_prompt(tokenizer, 32), 16)  # warmup: kernel selection and allocator growth
        for workload in workloads:
            metrics = measure(engine, make_prompt(tokenizer, workload.prompt_len), workload, args.repeats)
            rows.append((name, workload.name, metrics))
            print(f"{name:20s} {workload.name:14s} {metrics.decode_tok_s:6.1f} tok/s", flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + markdown_table(rows))
    save_json(args.out, vars(args) | {"out": str(args.out)}, rows)


if __name__ == "__main__":
    main()
