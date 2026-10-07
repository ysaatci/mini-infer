import torch
from torch import Tensor

from mini_infer.model import Qwen2ForCausalLM
from mini_infer.paged_cache import PagedBatch, PagedKVPool

# Batch sizes with a recorded graph. A step runs in the smallest bucket that fits, padded with dummy rows.
BATCH_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64)


class DecodeGraphRunner:
    """Replays a recorded forward step instead of launching hundreds of small kernels from Python.

    Launching a kernel costs about as much CPU time as a small decode kernel takes on the GPU, so a
    step was mostly launch overhead. A CUDA graph records the whole step once and replays it with one
    call. A graph freezes shapes and memory addresses, so inputs go through fixed buffers: each step
    copies the real token ids, positions and block tables in, pads the batch up to its bucket with
    dummy rows, replays, and reads logits from the graph's own output buffer.

    query_len is the number of new tokens per sequence: 1 for decode, k + 1 for speculative verification.
    """

    def __init__(self, model: Qwen2ForCausalLM, pool: PagedKVPool, max_batch_size: int, query_len: int = 1):
        self.model = model
        self.pool = pool
        self.query_len = query_len
        self.buckets = [b for b in BATCH_BUCKETS if b < max_batch_size] + [max_batch_size]
        # Dummy rows write their query_len tokens' k/v into this block, never into a real sequence's.
        if query_len > pool.block_size:
            raise ValueError("padding rows must fit in one block")
        padding_seq = f"__padding_{id(self)}__"  # one per runner: several runners can share a pool
        if not pool.reserve(padding_seq, query_len):
            raise RuntimeError("no free block for CUDA graph padding")
        self.padding_block = pool.tables[padding_seq][0]

        rows, device = self.buckets[-1], pool.k.device
        self.input_ids = torch.zeros(rows, query_len, dtype=torch.long, device=device)
        self.positions = torch.zeros(rows, query_len, dtype=torch.long, device=device)
        self.lengths = torch.zeros(rows, dtype=torch.int32, device=device)
        # Wide enough for one sequence holding every block.
        self.block_tables = torch.full((rows, pool.num_blocks), self.padding_block, dtype=torch.int32, device=device)

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.logits: dict[int, Tensor] = {}
        self._capture()

    def run(self, token_ids: list[list[int]], seq_ids: list[str]) -> Tensor:
        """query_len new tokens per sequence -> logits [B, query_len, vocab]. Valid until the next call.
        The caller advances the pool's lengths afterwards."""
        B, T = len(seq_ids), self.query_len
        bucket = next(b for b in self.buckets if b >= B)
        lengths = [self.pool.lengths[s] for s in seq_ids]  # the first new token's position = tokens already stored
        tables = self.pool.block_table_rows(seq_ids)
        offsets = torch.arange(T)

        self.input_ids[:B].copy_(torch.tensor(token_ids), non_blocking=True)
        self.positions[:B].copy_(torch.tensor(lengths)[:, None] + offsets, non_blocking=True)
        self.lengths[:B].copy_(torch.tensor(lengths, dtype=torch.int32), non_blocking=True)
        self.block_tables[:B, : len(tables[0])].copy_(torch.tensor(tables, dtype=torch.int32), non_blocking=True)
        # Padding rows: tokens 0 at positions 0.., stored in the padding block.
        self.input_ids[B:bucket] = 0
        self.positions[B:bucket] = offsets.to(self.positions.device)
        self.lengths[B:bucket] = 0
        self.block_tables[B:bucket, 0] = self.padding_block

        self.graphs[bucket].replay()
        return self.logits[bucket][:B]

    @torch.inference_mode()
    def _capture(self) -> None:
        memory_pool = torch.cuda.graph_pool_handle()
        for bucket in reversed(self.buckets):  # largest first, so smaller graphs reuse its memory

            def step(b: int = bucket) -> Tensor:
                # A fresh PagedBatch each run: it memoizes write positions, and a memo from the warmup
                # would be baked into the graph as a constant instead of recomputed on every replay.
                cache = PagedBatch(self.pool, self.block_tables[:b], self.lengths[:b], has_past=True)
                # Decode only needs the last position's logits; verification needs every position's.
                return self.model(self.input_ids[:b], self.positions[:b], cache, last_token_only=self.query_len == 1)

            # Warm up on a side stream first: Triton compiles its kernels and the allocator settles,
            # neither of which may happen while recording.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    step()
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=memory_pool):
                self.logits[bucket] = step()
            self.graphs[bucket] = graph
