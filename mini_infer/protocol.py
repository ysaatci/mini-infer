"""OpenAI API wire format: request bodies (validated) and response bodies."""

import time
import uuid

from pydantic import BaseModel, ConfigDict, Field


class _GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # like OpenAI-compatible servers: unknown fields are ignored

    model: str
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)  # newer name for max_tokens
    temperature: float = Field(1.0, ge=0, le=2)  # OpenAI's defaults
    top_p: float = Field(1.0, gt=0, le=1)
    n: int = Field(1, ge=1, le=1)  # one completion per request
    stream: bool = False
    ignore_eos: bool = False  # extension (vLLM has it too): benchmarks need fixed output lengths
    seed: int | None = None  # same seed and prompt: same sampled output (fully, with --deterministic)

    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(_GenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1)


class CompletionRequest(_GenerationRequest):
    prompt: str | list[int]  # text, or token ids (what benchmark clients send)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def chat_completion(id: str, model: str, text: str, finish_reason: str, usage: dict) -> dict:
    return {
        "id": id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": usage,
    }


def chat_chunk(id: str, model: str, delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def completion(id: str, model: str, text: str, finish_reason: str | None, usage: dict | None = None) -> dict:
    """Both the full response and a streamed chunk: completions use the same shape for each."""
    body = {
        "id": id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def model_list(model: str) -> dict:
    return {"object": "list", "data": [{"id": model, "object": "model", "created": 0, "owned_by": "mini-infer"}]}
