import pytest
import torch
from transformers import AutoTokenizer

from mini_infer.paged_cache import PagedKVPool
from mini_infer.generate import generate
from mini_infer.loader import load_model, resolve_model_dir

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Write a short story about a lighthouse keeper who finds a message in a bottle."

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def model():
    return load_model(NAME, dtype=torch.float32)


@pytest.fixture(scope="module")
def input_ids():
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_dir(NAME))
    return tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()


@torch.no_grad()
def test_chunked_forward_matches_full_forward(model, input_ids):
    # Covers all three attention paths: prefill, several tokens after a prefix, one token after a prefix.
    full = model(input_ids)
    # Tiny blocks, so the chunks cross block boundaries.
    pool = PagedKVPool(model.config, num_blocks=8, block_size=4, device=input_ids.device, dtype=torch.float32)
    pool.reserve("0", 13)
    cache = pool.view(["0"])
    chunks = [input_ids[:, :8], input_ids[:, 8:12], input_ids[:, 12:13]]
    chunked = torch.cat([model(chunk, cache=cache) for chunk in chunks], dim=1)
    torch.testing.assert_close(chunked, full[:, :13], atol=1e-3, rtol=1e-3)


def test_cached_generation_matches_uncached(model, input_ids):
    with_cache = generate(model, input_ids, max_new_tokens=32, use_cache=True)
    without_cache = generate(model, input_ids, max_new_tokens=32, use_cache=False)
    assert torch.equal(with_cache, without_cache)
