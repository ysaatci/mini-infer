from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from mini_infer.paged_cache import PagedBatch


class AttentionBackend(Protocol):
    """Stores the new tokens' k/v in the cache (if any) and computes attention over everything so far.

    Shapes: q [B, num_heads, T, D], k/v [B, num_kv_heads, T, D] for the T new tokens -> [B, num_heads, T, D].
    The backend owns both steps because only it knows how to read the cache layout back.
    """

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, cache: PagedBatch | None, layer: int) -> Tensor: ...


class TorchPagedBackend:
    """Reference implementation in plain PyTorch: gathers each sequence's blocks into one padded tensor."""

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, cache: PagedBatch | None, layer: int) -> Tensor:
        if cache is not None:
            cache.write(layer, k, v)
        if cache is None or not cache.has_past:
            # Nothing cached before these tokens: causal attention over the new k/v directly, no reads needed.
            # enable_gqa lets several query heads share one k/v head without copying k/v.
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.attend_cached(q, cache, layer)

    def attend_cached(self, q: Tensor, cache: PagedBatch, layer: int) -> Tensor:
        """Attention of the new tokens' queries over all cached keys (new tokens already written)."""
        keys, values = gather_blocks(cache, layer)
        T = q.shape[2]
        # New token i of row b sits at position lengths[b] + i and may see keys up to there.
        visible = cache.lengths_t[:, None] + torch.arange(T, device=q.device)  # [B, T]
        mask = (torch.arange(keys.shape[2], device=q.device) <= visible[..., None])[:, None]  # [B, 1, T, S]
        if T == 1:
            return decode_with_mask(q, keys, values, mask)
        return F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=True)


def gather_blocks(cache: PagedBatch, layer: int) -> tuple[Tensor, Tensor]:
    """Copy each row's blocks, in table order, into k/v [B, kv_heads, num_blocks * block_size, D]."""
    tables = cache.block_tables.long()
    B, W = tables.shape
    k = cache.pool.k[layer][tables]  # [B, W, kv_heads, block_size, D]
    v = cache.pool.v[layer][tables]
    _, _, H, S, D = k.shape
    return k.transpose(1, 2).reshape(B, H, W * S, D), v.transpose(1, 2).reshape(B, H, W * S, D)


def decode_with_mask(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    """One new token per row, with a padding mask.

    SDPA with both a mask and enable_gqa falls back to a slow kernel (~20x slower measured). Instead,
    the query heads sharing a k/v head are folded into the sequence dimension: q [B, 12, 1, D] becomes
    [B, 2, 6, D], matching k/v's 2 heads, so no GQA is needed. All 6 share the row's mask, so it broadcasts.
    """
    B, H, _, D = q.shape
    folded = q.view(B, k.shape[1], H // k.shape[1], D)
    return F.scaled_dot_product_attention(folded, k, v, attn_mask=mask).reshape(B, H, 1, D)
