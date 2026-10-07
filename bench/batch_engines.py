"""Batching engines behind one interface. Imports are inside each class because the two engines live
in different venvs (vLLM pins its own torch), and each process only has one of them installed."""

from collections.abc import Callable
from typing import Protocol

from bench.batch_workload import BatchRequest

StepOutput = tuple[str, int, bool]  # request id, new tokens this step, finished


class BatchEngine(Protocol):
    def add(self, request: BatchRequest) -> None: ...

    def step(self) -> list[StepOutput]: ...

    def has_unfinished(self) -> bool: ...


class MiniInferBatchEngine:
    def __init__(self, model_name: str, max_batch_size: int, max_len: int):
        from mini_infer.engine import LLMEngine
        from mini_infer.loader import load_model

        # Same KV memory as one max_len slot per batch entry.
        self.engine = LLMEngine(load_model(model_name), max_batch_size * max_len // 16, max_batch_size)

    def add(self, request: BatchRequest) -> None:
        # No eos: every engine generates exactly output_len tokens.
        self.engine.add_request(request.prompt_ids, request.output_len, request_id=request.id)

    def step(self) -> list[StepOutput]:
        return [(out.request_id, 1, out.finished) for out in self.engine.step()]

    def has_unfinished(self) -> bool:
        return self.engine.has_unfinished()


class VllmBatchEngine:
    def __init__(self, model_name: str, max_batch_size: int, max_len: int):
        import os

        # Keep vLLM's engine in this process, so step() timing and torch memory stats cover it.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # FlashInfer's sampler compiles with nvcc at startup (not installed). The benchmark is greedy anyway.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import EngineArgs, LLMEngine

        args = EngineArgs(
            model=model_name,
            dtype="bfloat16",
            max_model_len=max_len,
            max_num_seqs=max_batch_size,  # same batch cap as ours, so the comparison is engine vs engine
            enable_prefix_caching=False,  # our engine has none yet
            gpu_memory_utilization=0.8,
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


BATCH_ENGINES: dict[str, Callable[[str, int, int], BatchEngine]] = {
    "mini-infer": MiniInferBatchEngine,
    "vllm": VllmBatchEngine,
}
