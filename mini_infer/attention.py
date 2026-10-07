from typing import Protocol

import torch.nn.functional as F
from torch import Tensor


class AttentionBackend(Protocol):
    """Computes attention from rotated q/k and v.

    Shapes: q [B, num_heads, T, D], k/v [B, num_kv_heads, T, D] -> [B, num_heads, T, D].
    The model only depends on this interface, so cache-aware backends can replace it later.
    """

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor: ...


class SdpaBackend:
    """PyTorch's fused attention kernel with a causal mask over the full sequence."""

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # enable_gqa lets several query heads share one k/v head without copying k/v.
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
