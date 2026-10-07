# mini-infer

A small LLM inference engine written from scratch in PyTorch. It serves Qwen2.5 models and implements the main serving optimizations one by one, with a benchmark after each.

Built to understand what inference servers like vLLM do, not to replace them.

## What's implemented

| Technique | What it fixes |
|---|---|
| KV cache | Stops recomputing past tokens every step |
| Continuous batching | GPU sits idle when decoding one request at a time |
| Paged KV cache | Fixed per-request memory slots waste most of the cache |
| Speculative decoding | Decoding is memory-bound, so a small model drafts and the big one verifies |
| int8 weights | Smaller weights are read faster each step |

## Results

To be filled in as each step lands (RTX 5050 8 GB, Qwen2.5-1.5B, bf16).

## Run

```bash
uv sync
python -m mini_infer.server --model Qwen/Qwen2.5-1.5B-Instruct
python bench/run.py
```

Runs on Linux or WSL2 with CUDA 12.8+.
