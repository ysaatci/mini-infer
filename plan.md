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

### 2. Generation loop, no cache, then KV cache (done)
- First version recomputes the whole sequence every token. Then add a contiguous KV cache so each step processes one token.
- Why: the no-cache version is the baseline that shows the cost (quadratic work). The cache is the first real optimization and the base for everything else.

### 3. Benchmark harness (done)
- Fixed prompt set, fixed output length. Report tokens/s, time to first token, p50/p99 latency, peak GPU memory. Run against HF `generate` too.
- Why: built now so every later step has a before/after. Without it, claims in the README are guesses.

### 4. Continuous batching (done)
- Scheduler with a waiting queue and a running batch. Finished requests leave and new ones join between decode steps, instead of waiting for the whole batch to finish.
- KV cache here is still one big preallocated slot per request.
- Why: a GPU decoding one request is mostly idle. Batching raises throughput, and continuous batching avoids the batch waiting on its slowest request.

### 5. Paged KV cache
- Split KV memory into 16-token blocks, give each request a block table, allocate blocks on demand and free them on finish. Preempt the newest request (free its blocks, re-prefill later) when memory runs out.
- Triton decode kernel reads blocks in place through the table (no gather), splitting long sequences across programs when the batch is small. A plain PyTorch gather backend is the reference the kernel is tested against.
- Benchmark at step 4's KV memory (1.41 GB) with the batch cap raised from 32 to 64.
- Why: fixed slots reserve max length per request, so most memory sits unused and limits batch size. Paging wastes at most one block per request, so more requests fit. Step 4 also showed the gather of scattered slots costs ~half of each decode step, which the kernel removes.

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
mini_infer/   config.py  layers.py  model.py  loader.py  sampling.py  generate.py
              paged_cache.py  attention.py  kernels.py  request.py  scheduler.py  engine.py
              later: speculative.py  server.py
bench/        single request: workload.py  engines.py  run.py
              batching: batch_workload.py  batch_engines.py  batch_run.py
              shared: metrics.py  report.py  results/*.json
tests/        test_logits.py  test_cache.py  test_engine.py  test_paged_attention.py
```

## Improvements found along the way

Ideas noted while building. Each needs a benchmark before/after to earn its place.

- **CUDA graphs for decode.** Step 2 decodes at ~40 tok/s on the 1.5B model. The GPU's memory bandwidth allows ~100. The gap is probably CPU overhead: hundreds of small kernel launches per token. Recording one decode step as a CUDA graph and replaying it removes that. vLLM does this: its batched inter-token latency is ~15 ms vs our ~35 ms. After step 5 it's the clearest remaining gap: a decode step costs ~30 ms whether the batch is 1 or 20, so the floor is fixed per-step overhead, not GPU work.
- **Done: paged attention kernel instead of gathering.** Once requests finish at different times their memory is scattered, and reading it as one batch copied every row's cache each layer: 30 ms of a 60 ms decode step at batch 20. The Triton kernel reads blocks in place. Decode step at batch 64: 123 → 56 ms; at batch 20: 44 → 31 ms.
- **Done: split sequences across programs for small batches.** One program per (sequence, k/v head) put a single request on 2 of the GPU's 20 cores. Small batches now split each sequence into chunks and merge the partial softmaxes in a second Triton kernel. Merging with ~9 PyTorch ops per layer cost more than it saved; one merge kernel fixed that.
- **Uninitialized KV memory.** Found in step 5: attention reads whole blocks, and masked-out slots get zero weight, but 0 × NaN is NaN. The pool is zeroed once at allocation.
- **Batched prefill.** Newly admitted requests prefill one forward each. Packing several short prompts into one forward would use the GPU better.
- **Done: logits for the last token only.** Prefill computed logits for every prompt token (151k vocab each) but generation only uses the last one. Step 3 showed TTFT 361 vs 339 ms (HF) and peak memory 3.79 vs 3.30 GB at a 2048-token prompt. In step 4 it became a real bug: a view kept each prefill's full logits alive, ~20 prefills in one step pushed peak memory to 8.1 GB and spilled into system RAM. Fixed with `last_token_only`.
- **Done: fold GQA groups for masked decode.** SDPA with a padding mask and `enable_gqa` fell back to a kernel ~20× slower (4.3 vs 0.2 ms per layer). Folding the 6 query heads per k/v head into the sequence dimension avoids GQA mode. Batched decode step: 178 → 29 ms.
- **Fused RoPE / RMSNorm kernels (Triton).** Each is several small elementwise ops today. Fusing them cuts memory reads and launches.
- **Chunked prefill.** A long prompt blocks every other request while it runs. Splitting it into chunks lets decode steps of other requests interleave (fits after step 4).
- **Prefix caching.** Requests sharing a system prompt could reuse its KV blocks instead of recomputing them (fits after step 5).
