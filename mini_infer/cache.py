from typing import NamedTuple, Protocol

import torch
from torch import Tensor

from mini_infer.config import ModelConfig


class CachedKV(NamedTuple):
    """All keys/values so far for one layer, as attention should read them."""

    k: Tensor  # [B, num_kv_heads, S, D]
    v: Tensor
    # None: every row has the same length, the backend applies the causal mask itself.
    # Otherwise a complete bool mask [B, 1, T, S], True where a query may attend (rows differ in length).
    mask: Tensor | None


class KVCache(Protocol):
    """Stores keys/values of past tokens so each step only computes the new ones."""

    @property
    def length(self) -> int:
        """Number of tokens already stored."""
        ...

    def update(self, layer: int, k: Tensor, v: Tensor) -> CachedKV:
        """Store new k/v [B, num_kv_heads, T, D] for a layer and return all k/v so far."""
        ...

    def advance(self, num_tokens: int) -> None:
        """Mark new tokens as stored. Called once per forward, after every layer has updated."""
        ...


class StaticKVCache:
    """One preallocated tensor per k and v, sized for max_len tokens.

    Writing into a fixed buffer avoids copying the whole cache every step (as torch.cat would),
    but reserves memory for max_len even if the sequence ends early.
    """

    def __init__(
        self, config: ModelConfig, batch_size: int, max_len: int, device: torch.device, dtype: torch.dtype
    ):
        shape = (config.num_layers, batch_size, config.num_kv_heads, max_len, config.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def update(self, layer: int, k: Tensor, v: Tensor) -> CachedKV:
        start, end = self._length, self._length + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"cache full: {end} tokens > max_len {self.max_len}")
        self.k[layer, :, :, start:end] = k
        self.v[layer, :, :, start:end] = v
        return CachedKV(self.k[layer, :, :, :end], self.v[layer, :, :, :end], mask=None)

    def advance(self, num_tokens: int) -> None:
        self._length += num_tokens
