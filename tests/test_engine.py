import pytest
import torch
from transformers import AutoTokenizer

from mini_infer.engine import LLMEngine
from mini_infer.generate import generate
from mini_infer.loader import load_model, resolve_model_dir

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "The tallest mountain in the world is",
    "Write a haiku about autumn leaves falling on a quiet river.",
    "def fibonacci(n):",
    "List three uses of copper in modern electronics and explain each one briefly.",
    "Water boils at",
]
OUTPUT_LENS = [20, 35, 12, 40, 25]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_batched_engine_matches_single_request_generation():
    model = load_model(NAME, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_dir(NAME))
    prompts = [tokenizer(p).input_ids for p in PROMPTS]

    expected = {
        str(i): generate(model, torch.tensor([ids], device="cuda"), n)[0].tolist()
        for i, (ids, n) in enumerate(zip(prompts, OUTPUT_LENS))
    }

    # Fewer slots than requests: forces requests to wait, join mid-run, and reuse freed slots.
    engine = LLMEngine(model, max_batch_size=3, max_len=128)
    for i, (ids, n) in enumerate(zip(prompts, OUTPUT_LENS)):
        engine.add_request(ids, n, request_id=str(i))
    actual: dict[str, list[int]] = {str(i): [] for i in range(len(PROMPTS))}
    while engine.has_unfinished():
        for out in engine.step():
            actual[out.request_id].append(out.token)

    assert actual == expected
