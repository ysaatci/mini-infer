import time
from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

from mini_infer.engine import LLMEngine
from mini_infer.generate import stream
from mini_infer.loader import load_model, resolve_model_dir
from mini_infer.paged_cache import DEFAULT_BLOCK_SIZE, PagedKVPool
from mini_infer.quant import quantize_model

KV_CACHE_BYTES = int(1.41e9)  # same budget as the batching benchmark


class Engine(Protocol):
    def run(self, prompt_ids: Tensor, max_new_tokens: int) -> list[float]:
        """Generate exactly max_new_tokens greedily. Returns seconds from start until each token reached the CPU."""
        ...


class MiniInferEngine:
    """The serving engine (paged cache, Triton kernel, optional CUDA graphs) with one request at a time."""

    def __init__(self, model_name: str, use_cuda_graphs: bool, int8: bool = False, deterministic: bool = False):
        model = quantize_model(load_model(model_name)) if int8 else load_model(model_name)
        num_blocks = PagedKVPool.blocks_for_memory(model.config, KV_CACHE_BYTES, DEFAULT_BLOCK_SIZE, torch.bfloat16)
        self.engine = LLMEngine(model, num_blocks, max_batch_size=1, use_cuda_graphs=use_cuda_graphs, deterministic=deterministic)

    def run(self, prompt_ids: Tensor, max_new_tokens: int) -> list[float]:
        start = time.perf_counter()
        self.engine.add_request(prompt_ids[0].tolist(), max_new_tokens)
        times = []
        while self.engine.has_unfinished():
            for _ in self.engine.step():  # step() returns once the token is on the CPU
                times.append(time.perf_counter() - start)
        return times


class NoCacheEngine:
    """The step 2 baseline: recompute the whole sequence for every token."""

    def __init__(self, model_name: str):
        self.model = load_model(model_name)

    def run(self, prompt_ids: Tensor, max_new_tokens: int) -> list[float]:
        start = time.perf_counter()
        times = []
        for token in stream(self.model, prompt_ids.cuda(), max_new_tokens, use_cache=False):
            token.cpu()  # waits for the GPU, as a server must before sending the token
            times.append(time.perf_counter() - start)
        return times


class _TimingStreamer(BaseStreamer):
    def __init__(self, start: float):
        self.start = start
        self.times: list[float] = []
        self._seen_prompt = False

    def put(self, value: Tensor) -> None:
        if not self._seen_prompt:  # HF passes the prompt first
            self._seen_prompt = True
            return
        value.cpu()
        self.times.append(time.perf_counter() - self.start)

    def end(self) -> None:
        pass


class HuggingFaceEngine:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            resolve_model_dir(model_name), dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda()

    @torch.inference_mode()
    def run(self, prompt_ids: Tensor, max_new_tokens: int) -> list[float]:
        streamer = _TimingStreamer(time.perf_counter())
        self.model.generate(
            prompt_ids.cuda(),
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,  # no early stop, so every engine produces the same count
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            streamer=streamer,
        )
        return streamer.times


# Factories, so only one engine's weights sit on the GPU at a time.
ENGINES: dict[str, Callable[[str], Engine]] = {
    "mini-infer": lambda name: MiniInferEngine(name, use_cuda_graphs=True),
    "mini-infer-eager": lambda name: MiniInferEngine(name, use_cuda_graphs=False),
    "mini-infer-int8": lambda name: MiniInferEngine(name, use_cuda_graphs=True, int8=True),
    "mini-infer-det": lambda name: MiniInferEngine(name, use_cuda_graphs=True, deterministic=True),
    "mini-infer-nocache": NoCacheEngine,
    "hf": HuggingFaceEngine,
}
