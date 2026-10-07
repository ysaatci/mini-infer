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
    if all(p.temperature == 0 for p in params):
        return logits.argmax(-1)  # skip building distributions over the whole vocabulary
    return torch.multinomial(probabilities(logits, params), 1, generator=generator).squeeze(-1)


def probabilities(logits: Tensor, params: list[SamplingParams]) -> Tensor:
    """logits [B, vocab] -> the distribution each row samples from [B, vocab], float32.

    Temperature and top-p applied. Greedy rows are one-hot on their top token, so greedy is just a
    special case of sampling, which speculative decoding's acceptance rule relies on.
    """
    one_hot = torch.zeros_like(logits, dtype=torch.float32).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    if all(p.temperature == 0 for p in params):
        return one_hot  # skip the softmax and the vocabulary-wide sort of top-p
    temperature = logits.new_tensor([p.temperature for p in params], dtype=torch.float32)
    top_p = logits.new_tensor([p.top_p for p in params], dtype=torch.float32)
    greedy = temperature == 0
    # Greedy rows get temperature 1 here only to avoid dividing by zero; they're replaced below.
    probs = torch.softmax(logits.float() / temperature.masked_fill(greedy, 1)[:, None], dim=-1)
    probs = _top_p_filter(probs, top_p)
    return torch.where(greedy[:, None], one_hot, probs)


def _top_p_filter(probs: Tensor, top_p: Tensor) -> Tensor:
    """Keep, per row, the smallest set of most likely tokens whose total probability reaches top_p [B]."""
    sorted_probs, order = probs.sort(dim=-1, descending=True)
    # A token is dropped if the tokens ranked above it already reach top_p. The top token always stays.
    drop = sorted_probs.cumsum(-1) - sorted_probs >= top_p[:, None]
    sorted_probs = sorted_probs.masked_fill(drop, 0)
    filtered = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
    return filtered / filtered.sum(-1, keepdim=True)
