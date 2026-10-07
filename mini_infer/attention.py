from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


class AttentionBackend(Protocol):
    """Computes attention from rotated q/k and v.

    Shapes: q [B, num_heads, T, D] for the T new tokens, k/v [B, num_kv_heads, S, D] for all
    S tokens so far (S >= T, the new tokens are the last T). Returns [B, num_heads, T, D].
    The model only depends on this interface, so other backends can replace it later.
    """

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor: ...


class SdpaBackend:
    """PyTorch's fused attention kernel with a causal mask."""

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        T, S = q.shape[2], k.shape[2]
        # enable_gqa lets several query heads share one k/v head without copying k/v.
        if T == S:
            # Whole sequence at once (no cache, or first forward).
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        if T == 1:
            # One new token can see every earlier token, so no mask is needed.
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        # Several new tokens after a cached prefix. SDPA's is_causal aligns the mask to the top-left,
        # which is wrong here: new token i sits at position S - T + i and may see keys up to there.
        mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
