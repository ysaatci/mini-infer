import math

import torch
from torch import Tensor

from mini_infer.config import ModelConfig

DEFAULT_BLOCK_SIZE = 16  # vLLM's default: small enough that waste is low, large enough for efficient reads


class PagedKVPool:
    """KV memory split into fixed-size blocks. Each sequence owns a list of blocks (its block table).

    A sequence only holds blocks for tokens it actually has, so at most block_size - 1 slots per
    sequence are unused, instead of everything up to max_len as with one fixed slot per request.
    """

    def __init__(self, config: ModelConfig, num_blocks: int, block_size: int, device: torch.device, dtype: torch.dtype):
        # [layers, blocks, kv_heads, block_size, head_dim]: one (block, head) tile is contiguous, which is
        # what the attention kernel loads per step.
        shape = (config.num_layers, num_blocks, config.num_kv_heads, block_size, config.head_dim)
        # Zeros, not empty: attention reads whole blocks, including slots not yet written. Masking gives them
        # zero weight, but 0 * NaN is NaN, so uninitialized memory could poison the output.
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks))
        self.tables: dict[str, list[int]] = {}
        self.lengths: dict[str, int] = {}  # tokens written per sequence

    @staticmethod
    def blocks_for_memory(config: ModelConfig, memory_bytes: int, block_size: int, dtype: torch.dtype) -> int:
        """How many blocks fit in a KV memory budget (k and v, every layer)."""
        per_block = 2 * config.num_layers * config.num_kv_heads * block_size * config.head_dim * dtype.itemsize
        return memory_bytes // per_block

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_needed(self, seq_id: str, new_tokens: int) -> int:
        """Extra blocks a sequence needs to hold new_tokens more tokens."""
        total = self.lengths.get(seq_id, 0) + new_tokens
        return max(0, math.ceil(total / self.block_size) - len(self.tables.get(seq_id, [])))

    def reserve(self, seq_id: str, new_tokens: int) -> bool:
        """Make room for new_tokens more tokens. All or nothing: returns False if there aren't enough blocks."""
        needed = self.blocks_needed(seq_id, new_tokens)
        if needed > len(self._free):
            return False
        self.tables.setdefault(seq_id, []).extend(self._free.pop() for _ in range(needed))
        self.lengths.setdefault(seq_id, 0)
        return True

    def free(self, seq_id: str) -> None:
        self._free.extend(self.tables.pop(seq_id, []))
        self.lengths.pop(seq_id, None)

    def advance(self, seq_ids: list[str], num_tokens: int) -> None:
        for s in seq_ids:
            self.lengths[s] += num_tokens

    def block_table_rows(self, seq_ids: list[str], width: int | None = None) -> list[list[int]]:
        """Block tables padded to a common width with block 0. Padding entries are never read:
        lengths stop the kernel first."""
        tables = [self.tables[s] for s in seq_ids]
        width = width or max(len(t) for t in tables)
        return [t + [0] * (width - len(t)) for t in tables]

    def view(self, seq_ids: list[str]) -> "PagedBatch":
        """Metadata for one forward over these sequences."""
        device = self.k.device
        lengths = [self.lengths[s] for s in seq_ids]
        return PagedBatch(
            self,
            block_tables=torch.tensor(self.block_table_rows(seq_ids), dtype=torch.int32, device=device),
            lengths_t=torch.tensor(lengths, dtype=torch.int32, device=device),
            has_past=any(lengths),
            seq_ids=seq_ids,
            lengths=lengths,
        )


class PagedBatch:
    """Per-forward metadata for the sequences in a batch: where their blocks are and how long they are.

    Usually built by PagedKVPool.view(). A CUDA graph builds it once over its own fixed buffers instead,
    without seq_ids: those buffers are refilled before every replay and nothing on the Python side changes.
    """

    def __init__(
        self,
        pool: PagedKVPool,
        block_tables: Tensor,  # [B, max_blocks] int32
        lengths_t: Tensor,  # [B] int32, tokens stored before this forward
        has_past: bool,
        seq_ids: list[str] | None = None,
        lengths: list[int] | None = None,
    ):
        self.pool = pool
        self.block_tables = block_tables
        self.lengths_t = lengths_t
        self.has_past = has_past
        self.seq_ids = seq_ids
        self.lengths = lengths
        self._write_slots: tuple[Tensor, Tensor] | None = None

    @property
    def length(self) -> int:
        if len(set(self.lengths)) != 1:
            raise ValueError("rows have different lengths, pass positions explicitly")
        return self.lengths[0]

    def write(self, layer: int, k: Tensor, v: Tensor) -> None:
        """Store k/v [B, kv_heads, T, D] of the new tokens, at positions lengths[i] .. lengths[i] + T - 1."""
        B, H, T, D = k.shape
        blocks, offsets = self._slots(T)
        # Advanced indices on dims 0 and 2 with a slice between: the indexed shape is [B*T, kv_heads, D].
        self.pool.k[layer][blocks, :, offsets] = k.transpose(1, 2).reshape(B * T, H, D)
        self.pool.v[layer][blocks, :, offsets] = v.transpose(1, 2).reshape(B * T, H, D)

    def _slots(self, T: int) -> tuple[Tensor, Tensor]:
        """Physical (block, offset) of each new token. Same for every layer, so computed once per forward."""
        if self._write_slots is None:
            positions = self.lengths_t[:, None] + torch.arange(T, device=self.lengths_t.device)  # [B, T]
            blocks = self.block_tables.gather(1, (positions // self.pool.block_size).long()).flatten()
            self._write_slots = blocks, (positions % self.pool.block_size).flatten()
        return self._write_slots

    def advance(self, num_tokens: int) -> None:
        self.pool.advance(self.seq_ids, num_tokens)
        self.lengths = [n + num_tokens for n in self.lengths]
        self.lengths_t += num_tokens
        self.has_past = True
        self._write_slots = None
