import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_infer.loader import load_model, resolve_model_dir

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
PROMPT = "The quick brown fox jumps over the lazy dog. In 1969, the first humans landed on the"


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("name", MODELS)
@torch.no_grad()
def test_logits_match_huggingface(name: str) -> None:
    # fp32 so any mismatch comes from our code, not bf16 rounding.
    model_dir = resolve_model_dir(name)
    input_ids = AutoTokenizer.from_pretrained(model_dir)(PROMPT, return_tensors="pt").input_ids.cuda()

    # One model on the GPU at a time: two fp32 copies of the 1.5B model don't fit in 8 GB.
    # Straight onto the GPU: an fp32 copy in system RAM (6 GB for 1.5B) can exhaust WSL's memory mid-suite.
    reference = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="cuda")
    expected = reference(input_ids).logits
    del reference
    free_gpu()

    ours = load_model(name, dtype=torch.float32)
    actual = ours(input_ids)
    del ours
    free_gpu()

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
