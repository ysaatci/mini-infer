"""How many tokens the draft model proposes each step (k; 0 means plain decode)."""

from collections import deque
from typing import Protocol

from mini_infer.cuda_graphs import BATCH_BUCKETS


class DraftPolicy(Protocol):
    def choose(self, seq_ids: list[str]) -> int:
        """k for the next decode step of these sequences."""
        ...

    def record_step(self, k: int, batch_size: int, seconds: float) -> None:
        """How long a decode step with this k and batch size took."""
        ...

    def record_acceptance(self, seq_id: str, accepted: int, drafted: int) -> None: ...

    def release(self, seq_id: str) -> None:
        """The sequence finished, was aborted or was preempted."""
        ...


class FixedDraftPolicy:
    """Always the same k. The step 7 behavior, kept for comparison."""

    def __init__(self, k: int):
        self.k = k

    def choose(self, seq_ids: list[str]) -> int:
        return self.k

    def record_step(self, k: int, batch_size: int, seconds: float) -> None:
        pass

    def record_acceptance(self, seq_id: str, accepted: int, drafted: int) -> None:
        pass

    def release(self, seq_id: str) -> None:
        pass


def expected_tokens(k: int, acceptance: float) -> float:
    """Tokens a request gains from one step with k drafts, if each draft is accepted with probability
    `acceptance` until the first rejection: 1 + a + a^2 + ... + a^k (the +1 is the target's own token)."""
    if acceptance >= 1:
        return k + 1
    return (1 - acceptance ** (k + 1)) / (1 - acceptance)


def batch_bucket(batch_size: int) -> int:
    """Step time depends on the CUDA graph bucket the batch is padded to, not its exact size."""
    return next((b for b in BATCH_BUCKETS if b >= batch_size), BATCH_BUCKETS[-1])


class AdaptiveDraftPolicy:
    """Picks the k with the most expected tokens per second for the current batch.

    For each candidate k: sum over requests of expected_tokens(k, that request's acceptance), divided by
    the measured time of a step with that k at this batch size. More drafts mean more expected tokens
    per step but longer steps, and how much longer grows with the batch: at high load k = 0 wins.

    Acceptance is tracked per request (code is more predictable than stories), starting from the global
    average. Step times are moving averages per (k, batch bucket), seeded by a calibration run and kept
    up to date as the engine runs, so the policy fits whatever GPU and models it runs on.

    Every explore_every-th decision tries a neighboring k instead of the best one. Acceptance is only
    observed while speculating, so without this a policy that settles on k = 0 would never notice that
    the requests had become predictable enough to speculate on.
    """

    def __init__(
        self, max_draft_tokens: int = 4, prior_acceptance: float = 0.6, smoothing: float = 0.2, explore_every: int = 10
    ):
        self.max_draft_tokens = max_draft_tokens
        self.smoothing = smoothing  # weight of the newest observation in each moving average
        self.explore_every = explore_every
        self._decisions_made = 0
        self.global_acceptance = prior_acceptance
        self.acceptance: dict[str, float] = {}
        self.step_seconds: dict[tuple[int, int], float] = {}  # (k, batch bucket) -> moving average
        self.force_k: int | None = None  # calibration pins k to measure each one
        self.decisions: deque[tuple[int, int]] = deque(maxlen=100_000)  # (batch size, k), for plots

    def choose(self, seq_ids: list[str]) -> int:
        if self.force_k is not None:
            return self.force_k
        bucket = batch_bucket(len(seq_ids))
        rates = [self.acceptance.get(s, self.global_acceptance) for s in seq_ids]
        best_k, best_throughput = 0, 0.0
        for k in range(self.max_draft_tokens + 1):
            seconds = self.step_seconds.get((k, bucket))
            if seconds is None:
                continue  # never measured: don't gamble on it
            throughput = sum(expected_tokens(k, a) for a in rates) / seconds
            if throughput > best_throughput:
                best_k, best_throughput = k, throughput
        self._decisions_made += 1
        if self._decisions_made % self.explore_every == 0:
            best_k = self._neighbor(best_k)
        self.decisions.append((len(seq_ids), best_k))
        return best_k

    def _neighbor(self, k: int) -> int:
        """Alternately one above and one below k, kept within 0 .. max_draft_tokens."""
        go_up = (self._decisions_made // self.explore_every) % 2 == 1
        if k == 0 or (go_up and k < self.max_draft_tokens):
            return k + 1
        return k - 1

    def record_step(self, k: int, batch_size: int, seconds: float) -> None:
        key = (k, batch_bucket(batch_size))
        previous = self.step_seconds.get(key)
        self.step_seconds[key] = seconds if previous is None else self._blend(previous, seconds)

    def record_acceptance(self, seq_id: str, accepted: int, drafted: int) -> None:
        rate = accepted / drafted
        self.acceptance[seq_id] = self._blend(self.acceptance.get(seq_id, self.global_acceptance), rate)
        self.global_acceptance = self._blend(self.global_acceptance, rate)

    def release(self, seq_id: str) -> None:
        self.acceptance.pop(seq_id, None)

    def _blend(self, old: float, new: float) -> float:
        return (1 - self.smoothing) * old + self.smoothing * new
