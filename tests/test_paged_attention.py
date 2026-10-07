import random

import pytest
import torch

from mini_infer.attention import TorchPagedBackend, TritonPagedBackend
from mini_infer.config import ModelConfig
from mini_infer.paged_cache import PagedKVPool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Qwen2.5-1.5B attention shape: 12 query heads sharing 2 k/v heads, head_dim 128.
CONFIG = ModelConfig(
    vocab_size=1, hidden_size=1536, intermediate_size=1, num_layers=1, num_heads=12, num_kv_heads=2,
    rms_norm_eps=1e-6, rope_theta=1e4, tie_word_embeddings=False,
)
# Small batch (kernel splits each sequence across programs): a single token, exactly one full block,
# one past a block boundary, a long sequence. Large batch: enough programs already, no split.
SMALL_BATCH = [1, 16, 17, 100, 700]
LARGE_BATCH = [random.Random(1).randint(1, 300) for _ in range(64)]


@pytest.mark.parametrize("seq_lens", [SMALL_BATCH, LARGE_BATCH], ids=["split", "no-split"])
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_triton_decode_matches_reference(seq_lens, dtype, tol):
    torch.manual_seed(0)
    pool = PagedKVPool(CONFIG, num_blocks=1400, block_size=16, device="cuda", dtype=dtype)
    random.Random(0).shuffle(pool._free)  # scattered, out-of-order blocks, as after many requests come and go
    pool.k.normal_()
    pool.v.normal_()
    seq_ids = [str(i) for i in range(len(seq_lens))]
    for seq_id, n in zip(seq_ids, seq_lens):
        pool.reserve(seq_id, n)
        pool.lengths[seq_id] = n - 1  # the new token's k/v is already written at position n - 1

    cache = pool.view(seq_ids)
    q = torch.randn(len(seq_lens), CONFIG.num_heads, 1, CONFIG.head_dim, device="cuda", dtype=dtype)
    expected = TorchPagedBackend().attend_cached(q, cache, layer=0)
    actual = TritonPagedBackend().attend_cached(q, cache, layer=0)
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
