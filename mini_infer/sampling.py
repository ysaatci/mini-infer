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
    """logits [B, vocab] -> next token ids [B]."""
    if params.temperature == 0:
        return logits.argmax(-1)

    probs = torch.softmax(logits.float() / params.temperature, dim=-1)
    if params.top_p < 1:
        probs = _top_p_filter(probs, params.top_p)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def _top_p_filter(probs: Tensor, top_p: float) -> Tensor:
    """Keep the smallest set of most likely tokens whose total probability reaches top_p."""
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    # A token is dropped if the tokens ranked above it already reach top_p. The top token always stays.
    drop = sorted_probs.cumsum(-1) - sorted_probs >= top_p
    sorted_probs = sorted_probs.masked_fill(drop, 0)
    filtered = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
    return filtered / filtered.sum(-1, keepdim=True)
