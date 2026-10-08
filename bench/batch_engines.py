"""Batching engines behind one interface. Imports are inside each class because the two engines live
in different venvs (vLLM pins its own torch), and each process only has one of them installed."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from bench.batch_workload import BatchRequest

StepOutput = tuple[str, int, bool]  # request id, new tokens this step, finished


class BatchEngine(Protocol):
    def add(self, request: BatchRequest) -> None: ...

    def step(self) -> list[StepOutput]: ...

    def has_unfinished(self) -> bool: ...

    def stats(self) -> dict:
        """Engine-specific counters worth recording with the results."""
        ...


@dataclass(frozen=True)
class EngineSettings:
    model: str
    max_batch_size: int
    kv_cache_bytes: int  # the same KV memory budget for every engine
    max_len: int  # longest prompt + output
    int8: bool = False  # int8 weights (mini-infer only)
    deterministic: bool = False  # batch-invariant mode (mini-infer only)


class MiniInferBatchEngine:
    def __init__(self, settings: EngineSettings):
        import torch

        from mini_infer.engine import LLMEngine
        from mini_infer.loader import load_model
        from mini_infer.paged_cache import DEFAULT_BLOCK_SIZE, PagedKVPool

        model = load_model(settings.model)
        if settings.int8:
            from mini_infer.quant import quantize_model

            model = quantize_model(model)
        num_blocks = PagedKVPool.blocks_for_memory(model.config, settings.kv_cache_bytes, DEFAULT_BLOCK_SIZE, torch.bfloat16)
        self.engine = LLMEngine(model, num_blocks, settings.max_batch_size, deterministic=settings.deterministic)

    def add(self, request: BatchRequest) -> None:
        # No eos: every engine generates exactly output_len tokens.
        self.engine.add_request(request.prompt_ids, request.output_len, request_id=request.id)

    def step(self) -> list[StepOutput]:
        return [(out.request_id, 1, out.finished) for out in self.engine.step()]

    def has_unfinished(self) -> bool:
        return self.engine.has_unfinished()

    def stats(self) -> dict:
        return {"preemptions": self.engine.scheduler.num_preemptions}


class VllmBatchEngine:
    def __init__(self, settings: EngineSettings):
        import os

        # Keep vLLM's engine in this process, so step() timing and torch memory stats cover it.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # FlashInfer's sampler compiles with nvcc at startup (not installed). The benchmark is greedy anyway.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import EngineArgs, LLMEngine

        args = EngineArgs(
            model=settings.model,
            dtype="bfloat16",
            max_model_len=settings.max_len,
            max_num_seqs=settings.max_batch_size,  # same batch cap as ours, so the comparison is engine vs engine
            kv_cache_memory_bytes=settings.kv_cache_bytes,  # same KV memory as ours
            gpu_memory_utilization=0.8,  # only its startup free-memory check now; the KV size above wins
            enable_prefix_caching=False,  # our engine has none yet
        )
        self.engine = LLMEngine.from_engine_args(args)
        self.seen: dict[str, int] = {}

    def add(self, request: BatchRequest) -> None:
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=request.output_len, temperature=0, ignore_eos=True)
        self.engine.add_request(request.id, {"prompt_token_ids": request.prompt_ids}, params)
        self.seen[request.id] = 0

    def step(self) -> list[StepOutput]:
        outputs = []
        for out in self.engine.step():
            total = len(out.outputs[0].token_ids)  # cumulative, so count what's new since last step
            outputs.append((out.request_id, total - self.seen[out.request_id], out.finished))
            self.seen[out.request_id] = total
        return outputs

    def has_unfinished(self) -> bool:
        return self.engine.has_unfinished_requests()

    def stats(self) -> dict:
        return {}  # vLLM's scheduler counters aren't exposed through LLMEngine


BATCH_ENGINES: dict[str, Callable[[EngineSettings], BatchEngine]] = {
    "mini-infer": MiniInferBatchEngine,
    "vllm": VllmBatchEngine,
}
