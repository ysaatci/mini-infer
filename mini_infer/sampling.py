from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0
    # Same seed and same prompt: same sampled tokens, whatever else is in the batch (with batch-invariant
    # kernels). None: random as usual.
    seed: int | None = None

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def sample(logits: Tensor, params: SamplingParams, generator: torch.Generator | None = None) -> Tensor:
    """logits [B, vocab] -> next token ids [B], same params for every row."""
    return sample_batch(logits, [params] * logits.shape[0], generator)


def sample_batch(
    logits: Tensor,
    params: list[SamplingParams],
    generator: torch.Generator | None = None,
    positions: list[int] | None = None,
) -> Tensor:
    """logits [B, vocab] -> next token ids [B], each row with its own params (requests in a batch differ).
    positions: index of the token being sampled per row, which seeded rows need."""
    if all(p.temperature == 0 for p in params):
        return logits.argmax(-1)  # skip building distributions over the whole vocabulary
    tokens = torch.multinomial(probabilities(logits, params), 1, generator=generator).squeeze(-1)
    for i, p in enumerate(params):
        if p.seed is not None and p.temperature > 0:
            tokens[i] = seeded_sample(logits[i : i + 1], p, positions[i])
    return tokens


def seeded_sample(logits: Tensor, params: SamplingParams, position: int) -> Tensor:
    """One row's token from randomness that depends only on (seed, position), never on the batch.

    Gumbel-max: argmax(log p + g) with g = -log(-log u) draws from p exactly, like multinomial. The
    distribution is built from this row alone, so softmax, top-p sort and cumsum always see the same
    shape and sum in the same order.
    """
    probs = probabilities(logits, [params])[0]
    generator = torch.Generator(device=logits.device).manual_seed((params.seed * 1_000_003 + position) % 2**63)
    u = torch.rand(probs.shape, device=logits.device, generator=generator).clamp_(min=1e-20)
    return (probs.log() - (-u.log()).log()).argmax()


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
