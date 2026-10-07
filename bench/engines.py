import time
from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

from mini_infer.generate import stream
from mini_infer.loader import load_model, resolve_model_dir


class Engine(Protocol):
    def run(self, prompt_ids: Tensor, max_new_tokens: int) -> list[float]:
        """Generate exactly max_new_tokens greedily. Returns seconds from start until each token reached the CPU."""
        ...


class MiniInferEngine:
    def __init__(self, model_name: str, use_cache: bool):
        self.model = load_model(model_name)
        self.use_cache = use_cache

    def run(self, prompt_ids: Tensor, max_new_tokens: int) -> list[float]:
        start = time.perf_counter()
        times = []
        for token in stream(self.model, prompt_ids.cuda(), max_new_tokens, use_cache=self.use_cache):
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
    "mini-infer": lambda name: MiniInferEngine(name, use_cache=True),
    "mini-infer-nocache": lambda name: MiniInferEngine(name, use_cache=False),
    "hf": HuggingFaceEngine,
}
