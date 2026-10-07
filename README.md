# mini-infer

A small LLM inference engine written from scratch in PyTorch. It serves Qwen2.5 models and implements the main serving optimizations one by one, with a benchmark after each.

Built to understand what inference servers like vLLM do, not to replace them.

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

Planned: int8 weights.

## Results

To be filled in at the end (RTX 5050 8 GB, Qwen2.5-1.5B, bf16). Raw numbers per step are in `bench/results/`.

## Run

Linux or WSL2 with an NVIDIA GPU and CUDA 12.8+.

```bash
bash scripts/setup_wsl.sh          # venv + GPU check
python scripts/download_models.py  # Qwen2.5 0.5B and 1.5B
python -m mini_infer.server        # OpenAI-compatible API on localhost:8000
                                   # --draft-model Qwen/Qwen2.5-0.5B-Instruct for speculative decoding
pytest
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct", messages=[{"role": "user", "content": "Hello"}]
)
```
