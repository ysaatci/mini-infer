"""OpenAI-compatible HTTP server.

python -m mini_infer.server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
"""

import argparse
import json
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, GenerationConfig

from mini_infer import protocol
from mini_infer.async_engine import AsyncEngine
from mini_infer.detokenizer import IncrementalDetokenizer
from mini_infer.engine import LLMEngine, TokenOutput
from mini_infer.loader import load_model, resolve_model_dir
from mini_infer.paged_cache import DEFAULT_BLOCK_SIZE, PagedKVPool
from mini_infer.protocol import ChatCompletionRequest, CompletionRequest
from mini_infer.sampling import SamplingParams
from mini_infer.speculative import SpeculativeConfig

Tokens = AsyncIterator[TokenOutput]


def create_app(engine: AsyncEngine, tokenizer, model_name: str, stop_ids: frozenset[int]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine.start()
        yield
        engine.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    token_limit = engine.engine.max_request_tokens

    def start(request: ChatCompletionRequest | CompletionRequest, prompt_ids: list[int]) -> Tokens:
        # Checked here, before a stream starts: once it has, the 200 status is already sent.
        budget = token_limit - len(prompt_ids)
        max_new_tokens = request.requested_max_tokens() or budget
        if budget < 1 or max_new_tokens > budget:
            raise HTTPException(400, f"prompt ({len(prompt_ids)} tokens) + max_tokens exceeds the {token_limit}-token limit")
        params = SamplingParams(request.temperature, request.top_p)
        return engine.generate(prompt_ids, max_new_tokens, params, frozenset() if request.ignore_eos else stop_ids)

    @app.get("/v1/models")
    async def models() -> dict:
        return protocol.model_list(model_name)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        messages = [m.model_dump() for m in request.messages]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
        tokens, id = start(request, prompt_ids), protocol.new_id("chatcmpl")

        if request.stream:

            async def chunks():
                yield protocol.chat_chunk(id, model_name, {"role": "assistant", "content": ""})
                async for text, finish_reason in _text_deltas(tokens, tokenizer):
                    if text:
                        yield protocol.chat_chunk(id, model_name, {"content": text})
                    if finish_reason:
                        yield protocol.chat_chunk(id, model_name, {}, finish_reason)

            return _sse(chunks())

        text, finish_reason, count = await _collect(tokens, tokenizer)
        return protocol.chat_completion(id, model_name, text, finish_reason, protocol.usage(len(prompt_ids), count))

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompt_ids = request.prompt if isinstance(request.prompt, list) else tokenizer(request.prompt).input_ids
        tokens, id = start(request, prompt_ids), protocol.new_id("cmpl")

        if request.stream:

            async def chunks():
                async for text, finish_reason in _text_deltas(tokens, tokenizer):
                    if text or finish_reason:
                        yield protocol.completion(id, model_name, text, finish_reason)

            return _sse(chunks())

        text, finish_reason, count = await _collect(tokens, tokenizer)
        return protocol.completion(id, model_name, text, finish_reason, protocol.usage(len(prompt_ids), count))

    return app


async def _text_deltas(tokens: Tokens, tokenizer) -> AsyncIterator[tuple[str, str | None]]:
    """(new text, finish reason) per generated token. aclosing() makes sure that if the client
    disconnects mid-stream, the token stream is closed right away, which aborts the request."""
    detokenizer = IncrementalDetokenizer(tokenizer)
    async with aclosing(tokens) as stream:
        async for output in stream:
            text = detokenizer.add(output.token)
            if output.finished:
                text += detokenizer.flush()
            yield text, output.finish_reason


async def _collect(tokens: Tokens, tokenizer) -> tuple[str, str, int]:
    parts, finish_reason = [], None
    async for text, finish_reason in _text_deltas(tokens, tokenizer):
        parts.append(text)
    return "".join(parts), finish_reason, len(parts)


def _sse(chunks: AsyncIterator[dict]) -> StreamingResponse:
    """Server-sent events, the format OpenAI streams in: one JSON object per `data:` line, then [DONE]."""

    async def lines():
        async for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(lines(), media_type="text/event-stream")


def build_app(
    model_name: str,
    kv_cache_gb: float,
    max_batch_size: int,
    cuda_graphs: bool = True,
    draft_model: str | None = None,
    num_draft_tokens: int = 2,
) -> FastAPI:
    model_dir = resolve_model_dir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    eos = GenerationConfig.from_pretrained(model_dir).eos_token_id  # Qwen: <|im_end|> and <|endoftext|>
    stop_ids = frozenset([eos] if isinstance(eos, int) else eos)

    model = load_model(model_name)
    num_blocks = PagedKVPool.blocks_for_memory(model.config, int(kv_cache_gb * 1e9), DEFAULT_BLOCK_SIZE, torch.bfloat16)
    speculative = SpeculativeConfig(load_model(draft_model), num_draft_tokens) if draft_model else None
    engine = LLMEngine(model, num_blocks, max_batch_size, use_cuda_graphs=cuda_graphs, speculative=speculative)
    engine.warmup()
    return create_app(AsyncEngine(engine), tokenizer, model_name, stop_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--kv-cache-gb", type=float, default=1.41)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--draft-model", help="enables speculative decoding, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-draft-tokens", type=int, default=2)
    args = parser.parse_args()
    app = build_app(
        args.model,
        args.kv_cache_gb,
        args.max_batch_size,
        cuda_graphs=not args.no_cuda_graphs,
        draft_model=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
