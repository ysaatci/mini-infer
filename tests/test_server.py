import threading
import time

import openai
import pytest
import torch
import uvicorn

from mini_infer.server import build_app

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8123
MESSAGES = [{"role": "user", "content": "Name three primary colors."}]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def server():
    """The real server on a local port, driven over HTTP by the official openai client."""
    app = build_app(MODEL, kv_cache_gb=0.3, max_batch_size=4)
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    while not uv.started:
        time.sleep(0.05)
    yield app
    uv.should_exit = True
    thread.join()


@pytest.fixture(scope="module")
def client(server):
    return openai.OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="unused")


def test_lists_the_model(client):
    assert [m.id for m in client.models.list()] == [MODEL]


def test_streamed_chat_matches_non_streamed(client):
    full = client.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=40, temperature=0)
    choice = full.choices[0]
    assert choice.message.content and choice.finish_reason in ("stop", "length")
    assert full.usage.total_tokens == full.usage.prompt_tokens + full.usage.completion_tokens

    chunks = list(client.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=40, temperature=0, stream=True))
    streamed = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert streamed == choice.message.content
    assert chunks[-1].choices[0].finish_reason == choice.finish_reason


def test_completion_from_token_ids_runs_to_max_tokens(client):
    # ignore_eos isn't in the openai client's signature, so it goes in extra_body like other extensions.
    result = client.completions.create(
        model=MODEL, prompt=[785, 6722, 315, 9625, 374], max_tokens=16, temperature=0, extra_body={"ignore_eos": True}
    )
    assert result.choices[0].finish_reason == "length"
    assert result.usage.completion_tokens == 16


def test_rejects_requests_longer_than_the_cache(client):
    with pytest.raises(openai.BadRequestError):
        client.completions.create(model=MODEL, prompt="hi", max_tokens=10_000_000)


def test_disconnect_mid_stream_frees_the_request(server, client):
    engine = server.state.engine.engine
    free_before = engine.pool.num_free_blocks
    # 4000 tokens takes far longer than the 5 s deadline, so passing means it was aborted, not finished.
    stream = client.chat.completions.create(
        model=MODEL, messages=MESSAGES, max_tokens=4000, stream=True, extra_body={"ignore_eos": True}
    )
    next(iter(stream))  # generation has started
    stream.close()  # client goes away

    deadline = time.time() + 5
    while engine.has_unfinished() and time.time() < deadline:
        time.sleep(0.05)
    assert not engine.has_unfinished()
    assert engine.pool.num_free_blocks == free_before
