import pytest
import torch
from transformers import AutoTokenizer

from mini_infer.engine import LLMEngine
from mini_infer.generate import generate
from mini_infer.loader import load_model, resolve_model_dir
from mini_infer.speculative import SpeculativeConfig

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


@pytest.fixture(scope="module")
def model():
    return load_model(NAME, dtype=torch.float32)


@pytest.fixture(scope="module")
def prompts():
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_dir(NAME))
    return [tokenizer(p).input_ids for p in PROMPTS]


@pytest.fixture(scope="module")
def expected(model, prompts):
    return {
        str(i): generate(model, torch.tensor([ids], device="cuda"), n)[0].tolist()
        for i, (ids, n) in enumerate(zip(prompts, OUTPUT_LENS))
    }


# 64 blocks fit everything. With 5 blocks (80 tokens, one goes to CUDA graph padding) the longest
# request still fits alone (55 tokens), but three running requests outgrow memory and get preempted.
@pytest.mark.parametrize(
    "num_blocks, cuda_graphs, expect_preemption",
    [(64, False, False), (64, True, False), (5, True, True)],
    ids=["eager", "graphs", "graphs-preemption"],
)
def test_batched_engine_matches_single_request_generation(model, prompts, expected, num_blocks, cuda_graphs, expect_preemption):
    # Batch cap below the request count: forces requests to wait, join mid-run, and reuse freed blocks.
    engine = LLMEngine(model, num_blocks=num_blocks, max_batch_size=3, use_cuda_graphs=cuda_graphs)
    for i, (ids, n) in enumerate(zip(prompts, OUTPUT_LENS)):
        engine.add_request(ids, n, request_id=str(i))
    actual: dict[str, list[int]] = {str(i): [] for i in range(len(PROMPTS))}
    while engine.has_unfinished():
        for out in engine.step():
            actual[out.request_id].append(out.token)

    assert actual == expected
    assert (engine.scheduler.num_preemptions > 0) == expect_preemption


@pytest.fixture(scope="module")
def noisy_draft():
    """The target with 10% multiplicative noise on its weights: it agrees with the target often but not
    always (~60% of drafts accepted), so both acceptance and rejection get exercised."""
    draft = load_model(NAME, dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for name, param in draft.named_parameters():
            if param.dim() == 2 and "embed" not in name:
                param.mul_(1 + 0.1 * torch.randn(param.shape, device=param.device, generator=generator))
    return draft


@pytest.mark.parametrize(
    "num_blocks, cuda_graphs, expect_preemption",
    [(64, False, False), (64, True, False), (7, True, True)],
    ids=["eager", "graphs", "graphs-preemption"],
)
def test_speculative_greedy_output_matches_the_target_alone(
    model, prompts, expected, noisy_draft, num_blocks, cuda_graphs, expect_preemption
):
    speculative = SpeculativeConfig(noisy_draft, num_draft_tokens=4)
    engine = LLMEngine(model, num_blocks=num_blocks, max_batch_size=3, use_cuda_graphs=cuda_graphs, speculative=speculative)
    for i, (ids, n) in enumerate(zip(prompts, OUTPUT_LENS)):
        engine.add_request(ids, n, request_id=str(i))
    actual: dict[str, list[int]] = {str(i): [] for i in range(len(PROMPTS))}
    while engine.has_unfinished():
        for out in engine.step():
            actual[out.request_id].append(out.token)

    assert actual == expected
    assert 0 < engine.speculative.acceptance_rate < 1
    assert (engine.scheduler.num_preemptions > 0) == expect_preemption
