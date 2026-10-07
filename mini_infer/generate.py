import torch
from torch import Tensor

from mini_infer.cache import StaticKVCache
from mini_infer.model import Qwen2ForCausalLM
from mini_infer.sampling import SamplingParams, sample


@torch.inference_mode()
def generate(
    model: Qwen2ForCausalLM,
    prompt_ids: Tensor,
    max_new_tokens: int,
    params: SamplingParams = SamplingParams(),
    eos_id: int | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """prompt_ids [B, T] -> new token ids [B, <= max_new_tokens]. Stops early once every row hits eos_id."""
    B, T = prompt_ids.shape
    weight = model.lm_head.weight
    cache = StaticKVCache(model.config, B, T + max_new_tokens, weight.device, weight.dtype) if use_cache else None

    tokens = prompt_ids
    next_input = prompt_ids  # prefill: the whole prompt in one forward
    finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
    for _ in range(max_new_tokens):
        # Without a cache the model must re-read the whole sequence to produce one token.
        logits = model(next_input if use_cache else tokens, cache=cache)[:, -1]
        next_token = sample(logits, params, generator)
        if eos_id is not None:
            next_token = next_token.masked_fill(finished, eos_id)
            finished |= next_token == eos_id
        tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        next_input = next_token[:, None]
        if finished.all():
            break
    return tokens[:, T:]
