from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def sample(logits: Tensor, params: SamplingParams, generator: torch.Generator | None = None) -> Tensor:
    """logits [B, vocab] -> next token ids [B], same params for every row."""
    return sample_batch(logits, [params] * logits.shape[0], generator)


def sample_batch(logits: Tensor, params: list[SamplingParams], generator: torch.Generator | None = None) -> Tensor:
    """logits [B, vocab] -> next token ids [B], each row with its own params (requests in a batch differ)."""
    greedy_ids = logits.argmax(-1)
    if all(p.temperature == 0 for p in params):
        return greedy_ids

    temperature = logits.new_tensor([p.temperature for p in params], dtype=torch.float32)
    top_p = logits.new_tensor([p.top_p for p in params], dtype=torch.float32)
    # Greedy rows get temperature 1 here only to avoid dividing by zero; their result is replaced below.
    probs = torch.softmax(logits.float() / temperature.masked_fill(temperature == 0, 1)[:, None], dim=-1)
    probs = _top_p_filter(probs, top_p)
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.where(temperature == 0, greedy_ids, sampled)


def _top_p_filter(probs: Tensor, top_p: Tensor) -> Tensor:
    """Keep, per row, the smallest set of most likely tokens whose total probability reaches top_p [B]."""
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    # A token is dropped if the tokens ranked above it already reach top_p. The top token always stays.
    drop = sorted_probs.cumsum(-1) - sorted_probs >= top_p[:, None]
    sorted_probs = sorted_probs.masked_fill(drop, 0)
    filtered = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
    return filtered / filtered.sum(-1, keepdim=True)
