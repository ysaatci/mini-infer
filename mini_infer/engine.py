import itertools
from dataclasses import dataclass

import torch
from torch import Tensor

from mini_infer.cuda_graphs import DecodeGraphRunner
from mini_infer.model import Qwen2ForCausalLM
from mini_infer.paged_cache import DEFAULT_BLOCK_SIZE, PagedKVPool
from mini_infer.request import Request
from mini_infer.sampling import SamplingParams, sample_batch
from mini_infer.scheduler import Scheduler


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
    runs one decode step for every running request.
    """

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        num_blocks: int,
        max_batch_size: int = 64,
        block_size: int = DEFAULT_BLOCK_SIZE,
        use_cuda_graphs: bool = True,
    ):
        weight = model.lm_head.weight
        self.model = model
        self.pool = PagedKVPool(model.config, num_blocks, block_size, weight.device, weight.dtype)
        # Decode steps replay recorded graphs. Prefill stays eager: prompt lengths vary too much to record.
        self.graphs = DecodeGraphRunner(model, self.pool, max_batch_size) if use_cuda_graphs else None
        self.scheduler = Scheduler(self.pool, max_batch_size)
        self._ids = itertools.count()

    def add_request(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        params: SamplingParams = SamplingParams(),
        stop_ids: frozenset[int] = frozenset(),
        request_id: str | None = None,
    ) -> str:
        capacity = self.pool.num_blocks * self.pool.block_size
        if len(prompt_ids) + max_new_tokens > capacity:
            # Could never run, even alone with the whole cache.
            raise ValueError(f"prompt + max_new_tokens exceeds KV cache capacity of {capacity} tokens")
        request_id = request_id or str(next(self._ids))
        self.scheduler.add(Request(request_id, list(prompt_ids), max_new_tokens, params, frozenset(stop_ids)))
        return request_id

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def abort(self, request_id: str) -> None:
        """Stop a request and free its KV blocks. No further outputs are produced for it."""
        self.scheduler.abort(request_id)

    @torch.inference_mode()
    def step(self) -> list[TokenOutput]:
        batch = self.scheduler.schedule()
        if batch.prefill:
            requests = batch.prefill
            logits = torch.cat([self._prefill(r) for r in requests])
        elif batch.decode:
            requests = batch.decode
            logits = self._decode(requests)
        else:
            return []
        tokens = sample_batch(logits, [r.params for r in requests]).tolist()  # one GPU sync per step
        return [self._append(r, t) for r, t in zip(requests, tokens)]

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
            logits = self.graphs.decode([r.output_ids[-1] for r in requests], seq_ids)
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
