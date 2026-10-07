import asyncio
import itertools
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mini_infer.engine import LLMEngine, TokenOutput
from mini_infer.sampling import SamplingParams


@dataclass(frozen=True)
class _Add:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    params: SamplingParams
    stop_ids: frozenset[int]
    loop: asyncio.AbstractEventLoop
    outputs: asyncio.Queue  # where this request's tokens go


@dataclass(frozen=True)
class _Abort:
    request_id: str


_SHUTDOWN = object()


class AsyncEngine:
    """Runs LLMEngine on a background thread, so the web server's event loop never waits on the GPU.

    Only that thread touches the engine. The server sends it commands (add, abort) through a
    thread-safe inbox, and each request gets its tokens back through its own asyncio queue. The
    engine itself stays single-threaded and needs no locks.
    """

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[str, _Add] = {}  # engine thread only
        self._ids = itertools.count()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._inbox.put(_SHUTDOWN)
        self._thread.join()

    async def generate(
        self, prompt_ids: list[int], max_new_tokens: int, params: SamplingParams, stop_ids: frozenset[int]
    ) -> AsyncIterator[TokenOutput]:
        """Yields the request's tokens as the engine produces them. Closing the iterator early
        (client disconnected) aborts the request and frees its memory."""
        request_id = f"req-{next(self._ids)}"
        outputs: asyncio.Queue = asyncio.Queue()
        self._inbox.put(_Add(request_id, prompt_ids, max_new_tokens, params, stop_ids, asyncio.get_running_loop(), outputs))
        finished = False
        try:
            while not finished:
                item = await outputs.get()
                if isinstance(item, Exception):
                    raise item
                finished = item.finished
                yield item
        finally:
            if not finished:
                self._inbox.put(_Abort(request_id))

    # --- engine thread ---

    def _run(self) -> None:
        while True:
            # Sleep on the inbox while idle. While requests are running, only take what has arrived.
            try:
                command = self._inbox.get(block=not self.engine.has_unfinished())
                while True:
                    if command is _SHUTDOWN:
                        return
                    self._handle(command)
                    command = self._inbox.get_nowait()
            except queue.Empty:
                pass
            if self.engine.has_unfinished():
                try:
                    outputs = self.engine.step()
                except Exception as error:  # don't leave every client waiting forever
                    self._fail_all(error)
                    raise
                for output in outputs:
                    self._send(output)

    def _handle(self, command: _Add | _Abort) -> None:
        if isinstance(command, _Abort):
            self.engine.abort(command.request_id)
            self._streams.pop(command.request_id, None)
            return
        try:
            self.engine.add_request(
                command.prompt_ids, command.max_new_tokens, command.params, command.stop_ids, command.request_id
            )
        except ValueError as error:  # e.g. longer than the KV cache could ever hold
            command.loop.call_soon_threadsafe(command.outputs.put_nowait, error)
            return
        self._streams[command.request_id] = command

    def _send(self, output: TokenOutput) -> None:
        stream = self._streams.get(output.request_id)
        if stream is None:
            return  # aborted after this step started
        stream.loop.call_soon_threadsafe(stream.outputs.put_nowait, output)
        if output.finished:
            del self._streams[output.request_id]

    def _fail_all(self, error: Exception) -> None:
        for stream in self._streams.values():
            stream.loop.call_soon_threadsafe(stream.outputs.put_nowait, error)
        self._streams.clear()
