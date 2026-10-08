import random

import pytest
import torch
from transformers import AutoTokenizer

from mini_infer.engine import LLMEngine
from mini_infer.loader import load_model, resolve_model_dir
from mini_infer.sampling import SamplingParams
from mini_infer.speculative import SpeculativeConfig

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET = "Explain how a refrigerator keeps food cold, step by step."
OTHERS = [
    "Write a poem about the sea.", "List ten prime numbers.", "What is a mutex?", "Describe Istanbul in one paragraph.",
    "def quicksort(arr):", "Why is the sky blue?", "Summarize the French Revolution.", "Name three sorting algorithms.",
]
NEW_TOKENS = 64

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(resolve_model_dir(NAME))


def target_output(engine: LLMEngine, tokenizer, params: SamplingParams, company: int, join_late: bool) -> list[int]:
    """The target prompt's tokens when run with `company` other requests, optionally joining mid-batch."""
    rng = random.Random(company)
    others = [tokenizer(rng.choice(OTHERS)).input_ids for _ in range(company)]
    for i, ids in enumerate(others):
        engine.add_request(ids, rng.randint(20, 100), SamplingParams(temperature=1.0), request_id=f"other-{i}")
    if join_late:
        for _ in range(company + 5):  # others prefill and start decoding first
            engine.step()
    engine.add_request(tokenizer(TARGET).input_ids, NEW_TOKENS, params, request_id="target")
    tokens = []
    while engine.has_unfinished():
        tokens += [out.token for out in engine.step() if out.request_id == "target"]
    return tokens


# bf16: where batch-dependent summation order actually changes results. (fp32 would hide it.)
@pytest.mark.parametrize(
    "params", [SamplingParams(), SamplingParams(temperature=0.8, top_p=0.95, seed=1234)], ids=["greedy", "seeded"]
)
def test_output_does_not_depend_on_the_batch(tokenizer, params):
    engine = LLMEngine(load_model(NAME), num_blocks=600, max_batch_size=32, deterministic=True)
    alone = target_output(engine, tokenizer, params, company=0, join_late=False)
    assert len(alone) == NEW_TOKENS
    assert target_output(engine, tokenizer, params, company=7, join_late=False) == alone
    assert target_output(engine, tokenizer, params, company=20, join_late=True) == alone


def test_greedy_speculation_matches_plain_greedy(tokenizer):
    plain = LLMEngine(load_model(NAME), num_blocks=600, max_batch_size=32, deterministic=True)
    expected = target_output(plain, tokenizer, SamplingParams(), company=0, join_late=False)
    del plain

    speculative = SpeculativeConfig(load_model(NAME), max_batch_size=8)
    engine = LLMEngine(load_model(NAME), num_blocks=600, max_batch_size=32, speculative=speculative, deterministic=True)
    engine.warmup()  # calibrates the adaptive policy, so it actually speculates
    assert target_output(engine, tokenizer, SamplingParams(), company=0, join_late=False) == expected
    assert target_output(engine, tokenizer, SamplingParams(), company=3, join_late=True) == expected
    assert engine.speculative.num_drafted > 0
