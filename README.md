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

Linux or WSL2 with an NVIDIA GPU and CUDA 12.8+.

```bash
bash scripts/setup_wsl.sh          # venv + GPU check
python scripts/download_models.py  # Qwen2.5 0.5B and 1.5B
python -m bench.run --out bench/results/run.json
pytest
```
