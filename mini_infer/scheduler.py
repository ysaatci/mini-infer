from collections import deque
from dataclasses import dataclass

from mini_infer.cache import SlotKVPool
from mini_infer.request import Request, RequestStatus


@dataclass(frozen=True)
class Batch:
    """What the next engine step runs. Exactly one of the lists is non-empty."""

    prefill: list[Request]
    decode: list[Request]


class Scheduler:
    """First come, first served. Prefill first: whenever a slot is free and a request waits, admit it.

    Admitting eagerly keeps the batch full (throughput) at the cost of pausing running requests
    for one step while the newcomers prefill.
    """

    def __init__(self, pool: SlotKVPool):
        self.pool = pool
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> Batch:
        admitted = []
        while self.waiting and self.pool.num_free:
            request = self.waiting.popleft()
            request.slot = self.pool.allocate()
            request.status = RequestStatus.RUNNING
            admitted.append(request)
        if admitted:
            self.running.extend(admitted)
            return Batch(prefill=admitted, decode=[])
        return Batch(prefill=[], decode=list(self.running))

    def finish(self, request: Request) -> None:
        self.pool.free(request.slot)
        self.running.remove(request)
        request.status = RequestStatus.FINISHED
        request.slot = None
