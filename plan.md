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

### 5. Paged KV cache (done)
- Split KV memory into 16-token blocks, give each request a block table, allocate blocks on demand and free them on finish. Preempt the newest request (free its blocks, re-prefill later) when memory runs out.
- Triton decode kernel reads blocks in place through the table (no gather), splitting long sequences across programs when the batch is small. A plain PyTorch gather backend is the reference the kernel is tested against.
- Benchmark at step 4's KV memory (1.41 GB) with the batch cap raised from 32 to 64.
- Why: fixed slots reserve max length per request, so most memory sits unused and limits batch size. Paging wastes at most one block per request, so more requests fit. Step 4 also showed the gather of scattered slots costs ~half of each decode step, which the kernel removes.

### 5b. CUDA graphs (done)
- Record one decode step per batch-size bucket (1, 2, 4, 8, 16, 24, 32, 48, 64) and replay it. Inputs go through fixed buffers; batches pad up to the bucket with dummy rows that write to a reserved scratch block. Prefill stays eager.
- Prerequisites: the caller advances the cache (no Python bookkeeping inside the forward), the kernel's split count depends only on batch size, `PagedBatch` can wrap existing buffers.
- Why: after step 5 a decode step cost ~30 ms whether the batch was 1 or 20, so the floor was CPU launch overhead, not GPU work.

### 6. OpenAI-compatible server (done)
- FastAPI `/v1/chat/completions`, `/v1/completions` (streaming and not) and `/v1/models`.
- The engine loop runs on a background thread; the event loop sends it add/abort commands through a thread-safe inbox and gets tokens back through per-request asyncio queues. A client disconnect aborts its request and frees its blocks.
- Incremental detokenizer so streamed text never splits a multi-byte character.
- Verified with the official `openai` client; HTTP load test with the same workloads as the engine benchmark.
- Why: makes the engine usable from existing clients and shows it works under concurrent load, not just in a script.

### 7. Speculative decoding (done)
- 0.5B drafts k tokens, 1.5B verifies them in one forward pass. Rejection sampling (Leviathan et al.) decides acceptance, so output follows the 1.5B's distribution exactly at any temperature; greedy is the one-hot case.
- Inside the engine and batched: the draft has its own paged cache that catches up lazily (new prompt, resumed after preemption, steps without speculation). Rejected tokens roll back by resetting lengths. Off above a batch-size threshold.
- The Triton kernel now takes T new tokens per sequence (causal among them), so verification runs in place and as a CUDA graph.
- Benchmark on real chat prompts, not the repeated passage, which a draft predicts unrealistically well.
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
              paged_cache.py  attention.py  kernels.py  cuda_graphs.py  request.py  scheduler.py  engine.py
              async_engine.py  detokenizer.py  protocol.py  server.py  speculative.py
bench/        single request: workload.py  engines.py  run.py
              batching: batch_workload.py  batch_engines.py  batch_run.py  http_run.py
              speculative: chat_prompts.json  spec_run.py
              shared: metrics.py  report.py  results/*.json
tests/        test_logits.py  test_cache.py  test_engine.py  test_paged_attention.py  test_detokenizer.py  test_server.py
              test_speculative_sampling.py
```

## Improvements found along the way

Ideas noted while building. Each needs a benchmark before/after to earn its place.

- **Done: CUDA graphs for decode (step 5b).** After step 5 a decode step cost ~30 ms whether the batch was 1 or 20: CPU launch overhead, not GPU work. Single request: 36 → 64 tok/s (HF: 33–39). Offline batching: 874 → 985 tok/s (vLLM 1238). At 1–2 req/s, inter-token latency now matches vLLM (13–16 ms).
- **Chunked prefill.** The remaining gap to vLLM is under heavy load: at 3 req/s our worst-case token latency is 318 ms vs vLLM's 172, because a long prompt's prefill pauses every running request. vLLM splits prompts into chunks and mixes them into decode steps.
- **Serving overhead (step 6).** Over HTTP, inter-token latency is 3–7 ms higher than calling the engine directly (rate 1: 13.2 → 16.2 ms, rate 2: 15.8 → 22.9 ms) and offline throughput drops 8% (985 → 903 tok/s). The per-token Python work (detokenizing, JSON, SSE, one cross-thread handoff per token) runs on the event loop, and both threads share the GIL, so it slows the engine thread too. Cheaper: hand off one batch of tokens per step instead of one call per token. Fuller fix: run the engine in its own process, which is why vLLM moved to that design.
- **Cheaper drafting (step 7).** With k = 4 a request gains 3.4 tokens per target pass, yet single-request speed only rises 1.4× (58 → 81 tok/s), because the 0.5B draft runs k + 1 forwards per step and, with 24 layers against the target's 28, isn't proportionally cheaper under CUDA graphs. Options: drop the extra forward that only stores the last draft's k/v (handle it on the next step instead), a much smaller draft, or prompt-lookup (n-gram) drafting with no draft model at all.
- **Sampled speculation overhead (step 7).** At temperature 0.7 the gain is half the greedy one (60 → 72 vs 58 → 81 tok/s) at a similar acceptance rate. Building full softmax + top-p distributions over 152k tokens for every drafted and verified position costs real time; a fused kernel or sampling only the needed entries would cut it.
- **Batch-1 kernel splitting.** A single request is split 20 ways; the merge seems to cost more than it saves (batch 1 step 17.3 ms vs batch 4 at 14.1 ms). Capping splits by sequence length should fix it.
- **Done: paged attention kernel instead of gathering.** Once requests finish at different times their memory is scattered, and reading it as one batch copied every row's cache each layer: 30 ms of a 60 ms decode step at batch 20. The Triton kernel reads blocks in place. Decode step at batch 64: 123 → 56 ms; at batch 20: 44 → 31 ms.
- **Done: split sequences across programs for small batches.** One program per (sequence, k/v head) put a single request on 2 of the GPU's 20 cores. Small batches now split each sequence into chunks and merge the partial softmaxes in a second Triton kernel. Merging with ~9 PyTorch ops per layer cost more than it saved; one merge kernel fixed that.
- **Uninitialized KV memory.** Found in step 5: attention reads whole blocks, and masked-out slots get zero weight, but 0 × NaN is NaN. The pool is zeroed once at allocation.
- **Batched prefill.** Newly admitted requests prefill one forward each. Packing several short prompts into one forward would use the GPU better.
- **Done: logits for the last token only.** Prefill computed logits for every prompt token (151k vocab each) but generation only uses the last one. Step 3 showed TTFT 361 vs 339 ms (HF) and peak memory 3.79 vs 3.30 GB at a 2048-token prompt. In step 4 it became a real bug: a view kept each prefill's full logits alive, ~20 prefills in one step pushed peak memory to 8.1 GB and spilled into system RAM. Fixed with `last_token_only`.
- **Done: fold GQA groups for masked decode.** SDPA with a padding mask and `enable_gqa` fell back to a kernel ~20× slower (4.3 vs 0.2 ms per layer). Folding the 6 query heads per k/v head into the sequence dimension avoids GQA mode. Batched decode step: 178 → 29 ms.
- **Fused RoPE / RMSNorm kernels (Triton).** Each is several small elementwise ops today. Fusing them cuts memory reads and launches.
- **Prefix caching.** Requests sharing a system prompt could reuse its KV blocks instead of recomputing them (fits after step 5).
