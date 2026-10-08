# mini-infer

A small LLM inference engine written from scratch in PyTorch and Triton. It serves Qwen2.5 models through an OpenAI-compatible API and implements the main serving optimizations one by one, with a benchmark after each.

Built to understand what inference servers like vLLM do, not to replace them.

<img src="docs/architecture.svg" alt="Request path: client, FastAPI server, engine thread, engine step with scheduler and speculative decoder, CUDA graphs, Qwen2 model with optional int8 weights, Triton paged-attention kernel over a paged KV cache." width="760">

## Results

RTX 5050 Laptop GPU (8 GB), Qwen2.5-1.5B-Instruct, bf16 unless noted. Raw numbers are in `bench/results/`; charts are generated from them with `python -m bench.charts`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/single-request-dark.svg">
  <img alt="Single-request decode speed: Hugging Face 40 tok/s; no KV cache 15; KV cache 43; paged cache and Triton kernel 36; CUDA graphs 65; int8 weights 119." src="docs/charts/single-request-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/batching-dark.svg">
  <img alt="Throughput with 200 requests at once: continuous batching 438 tok/s, paged cache and kernel 874, CUDA graphs 985, int8 1132, vLLM 0.31 1238. Median request time at 1, 2 and 3 requests/s: mini-infer bf16 4.4, 6.3, 9.4 s; int8 3.1, 4.2, 5.8 s; vLLM bf16 4.2, 5.5, 6.9 s." src="docs/charts/batching-light.svg">
</picture>

Speculative decoding helps a quiet server and hurts a busy one, so the number of drafted tokens is chosen each step from measured acceptance and step times. A fixed setting is wrong somewhere; the adaptive policy stays within about 10% of the best policy at every load.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/speculation-policies-dark.svg">
  <img alt="Output tokens/s versus plain decode at 1, 2, 4, 8 and 16 requests in flight. Greedy: adaptive +17, -4, 0, -11, +7%; fixed k = 2 +18, +1, -3, -25, -14%; fixed k = 4 +18, 0, -10, -27, -20%. Temperature 0.7: adaptive +4, +1, +7, -1, +2%; fixed k = 2 +13, 0, -3, -23, -26%; fixed k = 4 +11, -3, -8, -29, -70%." src="docs/charts/speculation-policies-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/charts/adaptive-trace-dark.svg">
  <img alt="Over 110 seconds, load goes from 0.3 to 4 and back to 0.3 requests/s. While quiet the policy drafts 2 to 4 tokens per step; while busy (up to about 15 requests in flight) it drops to 0. Median request time: adaptive 3.27 s, plain decode 3.38 s, fixed k = 2 3.56 s." src="docs/charts/adaptive-trace-light.svg">
</picture>

int8 weights cost almost nothing in quality:

| | bf16 | int8 |
|---|---|---|
| WikiText-2 perplexity | 10.81 | 10.84 |
| Greedy tokens matching bf16 | | 98% |
| Weight memory | 3.09 GB | 2.02 GB |

## What's implemented

| Technique | What it fixes |
|---|---|
| KV cache | Stops recomputing past tokens every step |
| Continuous batching | GPU sits idle when decoding one request at a time |
| Paged KV cache | Fixed per-request memory slots waste most of the cache |
| Triton paged-attention kernel | Reading scattered cache blocks by copying them cost half of each step |
| CUDA graphs | Launching hundreds of small kernels from Python dominated each step |
| OpenAI-compatible server | Works with existing clients, streams tokens, frees memory on disconnect |
| Speculative decoding | Every token costs a full read of the weights; a small model drafts and the big one checks several per read |
| Adaptive speculation | The best number of drafts depends on load; a cost model picks it every step |
| int8 weights | Decode is limited by reading the weights; half the bytes per weight, with a Triton kernel that converts in registers |

## Run

Linux or WSL2 with an NVIDIA GPU and CUDA 12.8+.

```bash
bash scripts/setup_wsl.sh          # venv + GPU check
python scripts/download_models.py  # Qwen2.5 0.5B and 1.5B
python -m mini_infer.server        # OpenAI-compatible API on localhost:8000
                                   # --draft-model Qwen/Qwen2.5-0.5B-Instruct for speculative decoding
                                   # --int8 for int8 weights
pytest
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct", messages=[{"role": "user", "content": "Hello"}]
)
```
