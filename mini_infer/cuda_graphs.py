import torch
from torch import Tensor

from mini_infer.model import Qwen2ForCausalLM
from mini_infer.paged_cache import PagedBatch, PagedKVPool

# Batch sizes with a recorded graph. A step runs in the smallest bucket that fits, padded with dummy rows.
BATCH_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64)
PADDING_SEQ = "__padding__"


class DecodeGraphRunner:
    """Replays a recorded decode step instead of launching hundreds of small kernels from Python.

    Launching a kernel costs about as much CPU time as a small decode kernel takes on the GPU, so a
    step was mostly launch overhead. A CUDA graph records the whole step once and replays it with one
    call. A graph freezes shapes and memory addresses, so inputs go through fixed buffers: each step
    copies the real token ids, positions and block tables in, pads the batch up to its bucket with
    dummy rows, replays, and reads logits from the graph's own output buffer.
    """

    def __init__(self, model: Qwen2ForCausalLM, pool: PagedKVPool, max_batch_size: int):
        self.model = model
        self.pool = pool
        self.buckets = [b for b in BATCH_BUCKETS if b < max_batch_size] + [max_batch_size]
        # Dummy rows write their k/v here, so they never touch a real sequence's blocks.
        if not pool.reserve(PADDING_SEQ, 1):
            raise RuntimeError("no free block for CUDA graph padding")
        self.padding_block = pool.tables[PADDING_SEQ][0]

        rows, device = self.buckets[-1], pool.k.device
        self.input_ids = torch.zeros(rows, 1, dtype=torch.long, device=device)
        self.positions = torch.zeros(rows, 1, dtype=torch.long, device=device)
        self.lengths = torch.zeros(rows, dtype=torch.int32, device=device)
        # Wide enough for one sequence holding every block.
        self.block_tables = torch.full((rows, pool.num_blocks), self.padding_block, dtype=torch.int32, device=device)

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.logits: dict[int, Tensor] = {}
        self._capture()

    def decode(self, token_ids: list[int], seq_ids: list[str]) -> Tensor:
        """One decode step for these sequences -> logits [B, vocab]. Valid until the next call."""
        B = len(seq_ids)
        bucket = next(b for b in self.buckets if b >= B)
        lengths = [self.pool.lengths[s] for s in seq_ids]  # a token's position = tokens already stored
        tables = self.pool.block_table_rows(seq_ids)

        self.input_ids[:B, 0].copy_(torch.tensor(token_ids), non_blocking=True)
        self.positions[:B, 0].copy_(torch.tensor(lengths), non_blocking=True)
        self.lengths[:B].copy_(torch.tensor(lengths, dtype=torch.int32), non_blocking=True)
        self.block_tables[:B, : len(tables[0])].copy_(torch.tensor(tables, dtype=torch.int32), non_blocking=True)
        # Padding rows: token 0 at position 0, one token long, stored in the padding block.
        self.input_ids[B:bucket] = 0
        self.positions[B:bucket] = 0
        self.lengths[B:bucket] = 0
        self.block_tables[B:bucket, 0] = self.padding_block

        self.graphs[bucket].replay()
        return self.logits[bucket][:B, -1]

    @torch.inference_mode()
    def _capture(self) -> None:
        memory_pool = torch.cuda.graph_pool_handle()
        for bucket in reversed(self.buckets):  # largest first, so smaller graphs reuse its memory

            def step(b: int = bucket) -> Tensor:
                # A fresh PagedBatch each run: it memoizes write positions, and a memo from the warmup
                # would be baked into the graph as a constant instead of recomputed on every replay.
                cache = PagedBatch(self.pool, self.block_tables[:b], self.lengths[:b], has_past=True)
                return self.model(self.input_ids[:b], self.positions[:b], cache, last_token_only=True)

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
