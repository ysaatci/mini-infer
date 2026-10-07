from collections import deque
from dataclasses import dataclass

from mini_infer.paged_cache import PagedKVPool
from mini_infer.request import Request, RequestStatus


@dataclass(frozen=True)
class Batch:
    """What the next engine step runs. Exactly one of the lists is non-empty."""

    prefill: list[Request]
    decode: list[Request]


class Scheduler:
    """First come, first served, prefill first, with memory handed out in blocks as requests grow.

    Admitting eagerly keeps the batch full (throughput) at the cost of pausing running requests for
    one step while newcomers prefill. When blocks run out mid-decode, the newest request is preempted:
    its blocks are freed and it goes back to the front of the queue, to be prefilled again later.
    """

    def __init__(self, pool: PagedKVPool, max_batch_size: int):
        self.pool = pool
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # in admission order, newest last
        self.num_preemptions = 0

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> Batch:
        admitted = self._admit()
        if admitted:
            return Batch(prefill=admitted, decode=[])
        self._reserve_decode_space()
        return Batch(prefill=[], decode=list(self.running))

    def finish(self, request: Request) -> None:
        self.pool.free(request.id)
        self.running.remove(request)
        request.status = RequestStatus.FINISHED

    def abort(self, request_id: str) -> None:
        """Drop a request wherever it is (e.g. its client disconnected). Unknown or finished ids are ignored."""
        for request in self.running:
            if request.id == request_id:
                self.finish(request)
                return
        for request in self.waiting:
            if request.id == request_id:
                self.waiting.remove(request)
                self.pool.free(request.id)  # a preempted request may still hold an empty table entry
                request.status = RequestStatus.FINISHED
                return

    def _admit(self) -> list[Request]:
        admitted = []
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting[0]
            needed = self.pool.blocks_needed(request.id, len(request.all_ids))
            # Keep one free block per running request, so admitting doesn't force an immediate preemption.
            if needed + len(self.running) > self.pool.num_free_blocks:
                break
            self.pool.reserve(request.id, len(request.all_ids))
            self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            admitted.append(request)
        return admitted

    def _reserve_decode_space(self) -> None:
        """Every running request needs room for one more token. Oldest requests get it first."""
        for request in list(self.running):
            if request.status is not RequestStatus.RUNNING:
                continue  # preempted earlier in this loop
            while not self.pool.reserve(request.id, 1):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is request:
                    break

    def _preempt(self, request: Request) -> None:
        # Recompute instead of saving the KV to CPU: re-running a prefill is fast, copying cache out is not.
        self.pool.free(request.id)
        self.running.remove(request)
        request.status = RequestStatus.WAITING
        self.waiting.appendleft(request)
        self.num_preemptions += 1
