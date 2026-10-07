import itertools
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from mini_infer.cuda_graphs import BATCH_BUCKETS, DecodeGraphRunner
from mini_infer.draft_policy import AdaptiveDraftPolicy
from mini_infer.model import Qwen2ForCausalLM
from mini_infer.paged_cache import DEFAULT_BLOCK_SIZE, PagedKVPool
from mini_infer.request import Request
from mini_infer.sampling import SamplingParams, sample_batch
from mini_infer.scheduler import Scheduler
from mini_infer.speculative import SpeculativeConfig, SpeculativeDecoder


@dataclass(frozen=True)
class TokenOutput:
    request_id: str
    token: int
    finish_reason: str | None  # "stop", "length", or None while the request continues

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class LLMEngine:
    """Continuous batching: requests join and leave the running batch between steps.

    Drive it with add_request() and step(). Each step either prefills newly admitted requests or
    runs one decode step for every running request. With speculative decoding configured, small
    batches decode speculatively instead and each request can gain up to k + 1 tokens per step.
    """

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        num_blocks: int,
        max_batch_size: int = 64,
        block_size: int = DEFAULT_BLOCK_SIZE,
        use_cuda_graphs: bool = True,
        speculative: SpeculativeConfig | None = None,
    ):
        self.model = model
        self.pool = PagedKVPool(model.config, num_blocks, block_size, model.device, model.dtype)
        # Decode steps replay recorded graphs. Prefill stays eager: prompt lengths vary too much to record.
        self.graphs = DecodeGraphRunner(model, self.pool, max_batch_size) if use_cuda_graphs else None
        self.speculative = SpeculativeDecoder(model, self.pool, speculative, use_cuda_graphs) if speculative else None
        # Longest prompt + output one request can have: every block left after the graphs' scratch blocks.
        self.max_request_tokens = self.pool.num_free_blocks * block_size
        self.scheduler = Scheduler(
            self.pool,
            max_batch_size,
            decode_lookahead=self._decode_lookahead,
            on_release=self.speculative.release if self.speculative else lambda request_id: None,
        )
        self._ids = itertools.count()

    def _decode_lookahead(self, requests: list[Request]) -> int:
        """Cache slots the next decode step needs per request: k + 1, where the draft policy picks k
        (0 = plain decode). Batches above the speculation limit always decode plainly."""
        spec = self.speculative
        if spec is None or len(requests) > spec.max_batch_size:
            return 1
        return min(spec.policy.choose([r.id for r in requests]), spec.max_draft_tokens) + 1

    def add_request(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        params: SamplingParams = SamplingParams(),
        stop_ids: frozenset[int] = frozenset(),
        request_id: str | None = None,
    ) -> str:
        if len(prompt_ids) + max_new_tokens > self.max_request_tokens:
            # Could never run, even alone with the whole cache.
            raise ValueError(f"prompt + max_new_tokens exceeds KV cache capacity of {self.max_request_tokens} tokens")
        request_id = request_id or str(next(self._ids))
        self.scheduler.add(Request(request_id, list(prompt_ids), max_new_tokens, params, frozenset(stop_ids)))
        return request_id

    def warmup(self) -> None:
        """Run a few requests before serving. CUDA loads kernels lazily (~0.6 s measured for the first
        sampled request), and int8 matmuls tune their tiles per prompt-size class on first use; without
        this, real requests would pay for both."""
        prompt_lengths = [n for n in (8, 200, 600, 1500) if n + 4 <= self.max_request_tokens]
        for length in prompt_lengths:
            for params in (SamplingParams(), SamplingParams(temperature=1.0, top_p=0.9)):
                self.add_request([0] * length, 4, params, request_id="warmup")
                while self.has_unfinished():
                    self.step()
        if self.speculative is not None and isinstance(self.speculative.policy, AdaptiveDraftPolicy):
            self._calibrate_draft_policy(self.speculative.policy)

    def _calibrate_draft_policy(self, policy: AdaptiveDraftPolicy, steps: int = 10) -> None:
        """Time a few decode steps for every k at every batch bucket, so the policy starts with real costs
        instead of guesses. The engine keeps refining these while it serves."""
        buckets = [b for b in BATCH_BUCKETS if b <= self.speculative.max_batch_size]
        for batch_size in buckets:
            for k in range(self.speculative.max_draft_tokens + 1):
                ids = [f"calibrate-{i}" for i in range(batch_size)]
                if (64 + steps * (k + 1)) * batch_size > self.max_request_tokens:
                    continue  # too little cache memory to calibrate this size (tiny test engines)
                policy.calibrate(k)
                for request_id in ids:
                    self.add_request(list(range(1, 65)), steps * (k + 1), request_id=request_id)
                while self.has_unfinished():
                    self.step()
        policy.end_calibration()

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def abort(self, request_id: str) -> None:
        """Stop a request and free its KV blocks. No further outputs are produced for it."""
        self.scheduler.abort(request_id)

    @torch.inference_mode()
    def step(self) -> list[TokenOutput]:
        batch = self.scheduler.schedule()
        requests = batch.prefill or batch.decode
        if not requests:
            return []
        new_tokens = None
        if batch.decode and batch.lookahead > 1:
            new_tokens = self.speculative.step(requests, batch.lookahead - 1)  # None if the draft cache had no room
        if new_tokens is None:
            start = time.perf_counter()
            logits = torch.cat([self._prefill(r) for r in requests]) if batch.prefill else self._decode(requests)
            new_tokens = [[t] for t in sample_batch(logits, [r.params for r in requests]).tolist()]  # one GPU sync
            if batch.decode and self.speculative is not None:
                # The k = 0 baseline the draft policy weighs speculative steps against.
                self.speculative.policy.record_step(0, len(requests), time.perf_counter() - start)

        outputs = []
        for request, tokens in zip(requests, new_tokens):
            for token in tokens:
                outputs.append(self._append(request, token))
                if outputs[-1].finished:
                    break  # a stop token or max_new_tokens can land mid-way through accepted drafts
        return outputs

    def _prefill(self, request: Request) -> Tensor:
        # One request per forward: prompts differ in length, and padding them together wastes compute.
        # A preempted request re-runs its prompt plus what it had generated, then continues.
        input_ids = torch.tensor([request.all_ids], device=self.pool.k.device)
        cache = self.pool.view([request.id])
        logits = self.model(input_ids, cache=cache, last_token_only=True)[:, -1]
        cache.advance(input_ids.shape[1])
        return logits

    def _decode(self, requests: list[Request]) -> Tensor:
        if self.graphs is not None:
            seq_ids = [r.id for r in requests]
            logits = self.graphs.run([[r.output_ids[-1]] for r in requests], seq_ids)[:, -1]
            self.pool.advance(seq_ids, 1)
            return logits
        device = self.pool.k.device
        input_ids = torch.tensor([[r.output_ids[-1]] for r in requests], device=device)
        # A request's next position is the number of tokens it already has in the cache.
        positions = torch.tensor([[self.pool.lengths[r.id]] for r in requests], device=device)
        cache = self.pool.view([r.id for r in requests])
        logits = self.model(input_ids, positions, cache, last_token_only=True)[:, -1]
        cache.advance(1)
        return logits

    def _append(self, request: Request, token: int) -> TokenOutput:
        request.output_ids.append(token)
        finish_reason = request.finish_reason
        if finish_reason is not None:
            self.scheduler.finish(request)
        return TokenOutput(request.id, token, finish_reason)
