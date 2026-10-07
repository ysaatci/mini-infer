# mini-infer plan

Goal: a small LLM inference engine written from scratch, with measured numbers for each optimization.
Model: Qwen2.5-0.5B-Instruct (draft) and Qwen2.5-1.5B-Instruct (target), bf16, fits in 8 GB.
Stack: Python, PyTorch (cu128), FastAPI. Runs in WSL2 (Smart App Control blocks native wheels on Windows).
Rules: small modular commits (one logical change each), SOLID-style modules, benchmark after every optimization, tests only where a bug would be silent.

## Steps

### 0. Setup (done)
- WSL2 + CUDA 12.8 PyTorch, `uv` env, download both models.
- Why: everything after this depends on the GPU working. Fail early.

### 1. Own forward pass (done)
- Load safetensors weights into our own Qwen2 implementation: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP.
- Test (the only essential one): logits match Hugging Face within tolerance.
- Why: if we don't own the forward pass we can't change how attention reads the KV cache. Matching HF proves it's correct before we build on it.

### 2. Generation loop, no cache, then KV cache
- First version recomputes the whole sequence every token. Then add a contiguous KV cache so each step processes one token.
- Why: the no-cache version is the baseline that shows the cost (quadratic work). The cache is the first real optimization and the base for everything else.

### 3. Benchmark harness
- Fixed prompt set, fixed output length. Report tokens/s, time to first token, p50/p99 latency, peak GPU memory. Run against HF `generate` too.
- Why: built now so every later step has a before/after. Without it, claims in the README are guesses.

### 4. Continuous batching
- Scheduler with a waiting queue and a running batch. Finished requests leave and new ones join between decode steps, instead of waiting for the whole batch to finish.
- KV cache here is still one big preallocated slot per request.
- Why: a GPU decoding one request is mostly idle. Batching raises throughput, and continuous batching avoids the batch waiting on its slowest request.

### 5. Paged KV cache
- Split KV memory into fixed-size blocks, give each request a block table, allocate blocks on demand and free them on finish. Attention gathers keys/values through the table.
- Measure memory wasted by step 4's fixed slots vs paged.
- Why: fixed slots reserve max length per request, so most memory sits unused and limits batch size. Paging wastes at most one block per request, so more requests fit.

### 6. OpenAI-compatible server
- FastAPI `/v1/chat/completions` with streaming, requests feed the scheduler.
- Why: makes the engine usable from existing clients and shows it works under concurrent load, not just in a script.

### 7. Speculative decoding
- 0.5B drafts k tokens, 1.5B verifies them in one forward pass, accept the matching prefix.
- Report speedup and acceptance rate.
- Why: decoding is memory-bound, so checking k tokens costs about the same as generating one. If the draft is usually right we get several tokens per big-model pass.

### 8. int8 weight quantization (stretch)
- Per-channel int8 weights, dequantize in the matmul. Report memory, speed, and perplexity change.
- Why: weights are what decoding reads every step, so smaller weights mean faster decode. Perplexity shows what it costs.

### 9. README
- Results table and graphs from the benchmark harness, short explanation of each technique, how to run.
- Why: this is what a reviewer reads first.

## Layout

```
mini_infer/   model.py  cache.py  scheduler.py  sampling.py  speculative.py  server.py
bench/        run.py  prompts.json
tests/        test_logits.py
```
