import itertools
from dataclasses import dataclass

import torch
from torch import Tensor

from mini_infer.cache import SlotKVPool
from mini_infer.model import Qwen2ForCausalLM
from mini_infer.request import Request
from mini_infer.sampling import SamplingParams, sample_batch
from mini_infer.scheduler import Scheduler


@dataclass(frozen=True)
class TokenOutput:
    request_id: str
    token: int
    finished: bool


class LLMEngine:
    """Continuous batching: requests join and leave the running batch between steps.

    Drive it with add_request() and step(). Each step either prefills newly admitted requests or
    runs one decode step for every running request.
    """

    def __init__(self, model: Qwen2ForCausalLM, max_batch_size: int = 32, max_len: int = 2048):
        weight = model.lm_head.weight
        self.model = model
        self.pool = SlotKVPool(model.config, max_batch_size, max_len, weight.device, weight.dtype)
        self.scheduler = Scheduler(self.pool)
        self._ids = itertools.count()

    def add_request(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        params: SamplingParams = SamplingParams(),
        eos_id: int | None = None,
        request_id: str | None = None,
    ) -> str:
        if len(prompt_ids) + max_new_tokens > self.pool.max_len:
            raise ValueError(f"prompt + max_new_tokens exceeds max_len {self.pool.max_len}")
        request_id = request_id or str(next(self._ids))
        self.scheduler.add(Request(request_id, list(prompt_ids), max_new_tokens, params, eos_id))
        return request_id

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

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
        input_ids = torch.tensor([request.prompt_ids], device=self.pool.k.device)
        return self.model(input_ids, cache=self.pool.view([request.slot]))[:, -1]

    def _decode(self, requests: list[Request]) -> Tensor:
        device = self.pool.k.device
        input_ids = torch.tensor([[r.output_ids[-1]] for r in requests], device=device)
        # A request's next position is the number of tokens it already has in the cache.
        positions = torch.tensor([[self.pool.lengths[r.slot]] for r in requests], device=device)
        return self.model(input_ids, positions, self.pool.view([r.slot for r in requests]))[:, -1]

    def _append(self, request: Request, token: int) -> TokenOutput:
        request.output_ids.append(token)
        finished = request.is_finished
        if finished:
            self.scheduler.finish(request)
        return TokenOutput(request.id, token, finished)
