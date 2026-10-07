import math
from collections.abc import Iterator

import torch
from torch import Tensor

from mini_infer.paged_cache import DEFAULT_BLOCK_SIZE, PagedKVPool
from mini_infer.model import Qwen2ForCausalLM
from mini_infer.sampling import SamplingParams, sample


@torch.inference_mode()
def stream(
    model: Qwen2ForCausalLM,
    prompt_ids: Tensor,
    max_new_tokens: int,
    params: SamplingParams = SamplingParams(),
    eos_id: int | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """prompt_ids [B, T] -> yields next token ids [B] one step at a time. Stops once every row hits eos_id."""
    B, T = prompt_ids.shape
    cache = None
    if use_cache:
        # No scheduler here: each row reserves its full length up front.
        seq_ids = [str(i) for i in range(B)]
        num_blocks = B * math.ceil((T + max_new_tokens) / DEFAULT_BLOCK_SIZE)
        pool = PagedKVPool(model.config, num_blocks, DEFAULT_BLOCK_SIZE, model.device, model.dtype)
        for seq_id in seq_ids:
            pool.reserve(seq_id, T + max_new_tokens)
        cache = pool.view(seq_ids)

    tokens = prompt_ids
    next_input = prompt_ids  # prefill: the whole prompt in one forward
    finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
    for _ in range(max_new_tokens):
        # Without a cache the model must re-read the whole sequence to produce one token.
        logits = model(next_input if use_cache else tokens, cache=cache, last_token_only=True)[:, -1]
        if cache is not None:
            cache.advance(next_input.shape[1])
        next_token = sample(logits, params, generator)
        if eos_id is not None:
            next_token = next_token.masked_fill(finished, eos_id)
            finished |= next_token == eos_id
        yield next_token
        next_input = next_token[:, None]
        if not use_cache:
            tokens = torch.cat([tokens, next_input], dim=1)
        if finished.all():
            break


def generate(model: Qwen2ForCausalLM, prompt_ids: Tensor, max_new_tokens: int, **kwargs) -> Tensor:
    """Collects stream() into new token ids [B, <= max_new_tokens]."""
    return torch.stack(list(stream(model, prompt_ids, max_new_tokens, **kwargs)), dim=1)
