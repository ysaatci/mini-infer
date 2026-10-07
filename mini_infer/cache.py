import heapq
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


class SlotKVPool:
    """A fixed number of slots, each preallocated for max_len tokens. A request holds one slot for its lifetime.

    Writing into fixed buffers avoids copying the cache every step (as torch.cat would), but every
    slot reserves memory for max_len tokens even when its request is much shorter.
    """

    def __init__(self, config: ModelConfig, num_slots: int, max_len: int, device: torch.device, dtype: torch.dtype):
        shape = (config.num_layers, num_slots, config.num_kv_heads, max_len, config.head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.lengths = [0] * num_slots
        self._free = list(range(num_slots))  # min-heap: lowest slots first keeps batches contiguous

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("no free slots")
        slot = heapq.heappop(self._free)
        self.lengths[slot] = 0
        return slot

    def free(self, slot: int) -> None:
        heapq.heappush(self._free, slot)

    def view(self, slots: list[int]) -> "SlotBatch":
        return SlotBatch(self, slots)


class SlotBatch:
    """The KVCache seen by one forward pass: the slots of the requests in the batch, one row each."""

    def __init__(self, pool: SlotKVPool, slots: list[int]):
        self.pool = pool
        self.slots = slots
        device = pool.k.device
        self.slots_t = torch.tensor(slots, device=device)
        first = slots[0]
        # Consecutive slots can be sliced (a view, no copy). Others need a gather.
        if slots == list(range(first, first + len(slots))):
            self.index: slice | Tensor = slice(first, first + len(slots))
        else:
            self.index = self.slots_t
        self.lengths = [pool.lengths[s] for s in slots]
        self.lengths_t = torch.tensor(self.lengths, device=device)
        self._mask: Tensor | None = None

    @property
    def length(self) -> int:
        if len(set(self.lengths)) != 1:
            raise ValueError("rows have different lengths, pass positions explicitly")
        return self.lengths[0]

    def update(self, layer: int, k: Tensor, v: Tensor) -> CachedKV:
        T = k.shape[2]
        if max(self.lengths) + T > self.pool.max_len:
            raise ValueError(f"cache full: {max(self.lengths) + T} tokens > max_len {self.pool.max_len}")
        k_layer, v_layer = self.pool.k[layer], self.pool.v[layer]  # [num_slots, H, max_len, D]

        if len(set(self.lengths)) == 1:
            start = self.lengths[0]
            k_layer[self.index, :, start : start + T] = k
            v_layer[self.index, :, start : start + T] = v
            end = start + T
            return CachedKV(k_layer[self.index, :, :end], v_layer[self.index, :, :end], mask=None)

        if T != 1:
            raise ValueError("rows of different lengths can only add one token per forward")
        # Each row writes its token at its own position.
        k_layer[self.slots_t, :, self.lengths_t] = k[:, :, 0]
        v_layer[self.slots_t, :, self.lengths_t] = v[:, :, 0]
        # Read up to the longest row. Shorter rows are padded, the mask hides the padding.
        end = max(self.lengths) + 1
        return CachedKV(k_layer[self.index, :, :end], v_layer[self.index, :, :end], self._padding_mask(end))

    def _padding_mask(self, end: int) -> Tensor:
        # Same for every layer, so built once per forward. Row i may see keys 0..lengths[i] (its new token).
        if self._mask is None:
            keys = torch.arange(end, device=self.lengths_t.device)
            self._mask = (keys[None, :] <= self.lengths_t[:, None])[:, None, None, :]
        return self._mask

    def advance(self, num_tokens: int) -> None:
        for s in self.slots:
            self.pool.lengths[s] += num_tokens
        self.lengths = [n + num_tokens for n in self.lengths]
        self.lengths_t += num_tokens
        self._mask = None
