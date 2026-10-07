"""How many tokens the draft model proposes each step (k; 0 means plain decode)."""

import statistics
import time
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

    Acceptance is tracked per request (code is more predictable than stories) as counts shrunk toward
    the global rate: (accepted + w * global) / (drafted + w). One speculative step's acceptance is very
    noisy (k = 2 gives 0, 0.5 or 1), so a request only moves away from the global rate with evidence. A
    plain moving average let a few unlucky steps make a request look hopeless, and since acceptance is
    only observed while speculating, it never recovered.

    Step times are moving averages per (k, batch bucket), seeded by a calibration run and kept up to
    date as the engine runs, so the policy fits whatever GPU and models it runs on.

    Every explore_every-th decision tries the runner-up k instead of the best one, if it's predicted to
    be nearly as good. Acceptance is only observed while speculating, so without this a policy that
    settles on k = 0 would never notice that the requests had become predictable enough to speculate on.
    """

    def __init__(
        self,
        max_draft_tokens: int = 4,
        # Optimistic on purpose: a high guess costs a few speculative steps before measurements correct it,
        # a low guess can keep the policy from ever speculating, and so from ever learning it was wrong.
        prior_acceptance: float = 0.8,
        prior_weight: float = 16,
        smoothing: float = 0.2,
        explore_every: int = 10,
        explore_margin: float = 0.1,
    ):
        self.max_draft_tokens = max_draft_tokens
        self.prior_weight = prior_weight  # drafted tokens' worth of trust in the global rate
        self.smoothing = smoothing  # weight of the newest observation in each step-time moving average
        self.explore_every = explore_every
        self.explore_margin = explore_margin  # explore a k predicted within this fraction of the best
        self._decisions_made = 0
        self.global_acceptance = prior_acceptance
        self.counts: dict[str, tuple[int, int]] = {}  # per request: (accepted, drafted)
        self.step_seconds: dict[tuple[int, int], float] = {}  # (k, batch bucket) -> moving average
        self.force_k: int | None = None  # calibration pins k to measure each one
        self._calibration: dict[tuple[int, int], list[float]] = {}
        self.decisions: deque[tuple[float, int, int]] = deque(maxlen=100_000)  # (time, batch size, k), for plots

    def calibrate(self, k: int) -> None:
        """Pin k: the following steps measure its cost. Acceptance isn't recorded (synthetic prompts)."""
        self.force_k = k

    def end_calibration(self) -> None:
        """Seed each step time with the median of its calibration steps, minus the first. The first step
        of a new shape pays one-time setup, and seeding the average with it inflated estimates by up to ~35%."""
        for key, samples in self._calibration.items():
            self.step_seconds[key] = statistics.median(samples[1:] or samples)
        self._calibration.clear()
        self.force_k = None

    def choose(self, seq_ids: list[str]) -> int:
        if self.force_k is not None:
            return self.force_k
        throughput = self.predicted_throughput(seq_ids)
        ranked = sorted(throughput, key=throughput.get, reverse=True) or [0]
        chosen = ranked[0]
        self._decisions_made += 1
        if self._decisions_made % self.explore_every == 0 and len(ranked) > 1:
            # Re-check the runner-up if it would be a near-tie for a typical request (global acceptance).
            # Judging by these requests' own estimates would keep an unluckily pessimistic request at
            # k = 0 forever. A clearly worse k (more drafts on a busy GPU) is still never explored.
            runner_up = ranked[1]
            typical = self.predicted_throughput(seq_ids, typical=True)
            if typical[runner_up] >= (1 - self.explore_margin) * throughput[chosen]:
                chosen = runner_up
        self.decisions.append((time.perf_counter(), len(seq_ids), chosen))
        return chosen

    def predicted_throughput(self, seq_ids: list[str], typical: bool = False) -> dict[int, float]:
        """Expected tokens per second for each k with a measured step time at this batch size.
        typical: as if every request had the global acceptance rate."""
        bucket = batch_bucket(len(seq_ids))
        rates = [self.global_acceptance if typical else self.request_acceptance(s) for s in seq_ids]
        return {
            k: sum(expected_tokens(k, a) for a in rates) / seconds
            for k in range(self.max_draft_tokens + 1)
            if (seconds := self.step_seconds.get((k, bucket))) is not None  # never measured: don't gamble on it
        }

    def record_step(self, k: int, batch_size: int, seconds: float) -> None:
        key = (k, batch_bucket(batch_size))
        if self.force_k is not None:
            self._calibration.setdefault(key, []).append(seconds)
            return
        previous = self.step_seconds.get(key)
        self.step_seconds[key] = seconds if previous is None else self._blend(previous, seconds)

    def request_acceptance(self, seq_id: str) -> float:
        accepted, drafted = self.counts.get(seq_id, (0, 0))
        return (accepted + self.prior_weight * self.global_acceptance) / (drafted + self.prior_weight)

    def record_acceptance(self, seq_id: str, accepted: int, drafted: int) -> None:
        if self.force_k is not None:
            return  # calibration prompts are synthetic: their acceptance says nothing about real requests
        total_accepted, total_drafted = self.counts.get(seq_id, (0, 0))
        self.counts[seq_id] = (total_accepted + accepted, total_drafted + drafted)
        # Pooled over every request, so it can move slowly: a few drafted tokens shift it a little.
        weight = drafted / (drafted + self.prior_weight)
        self.global_acceptance += weight * (accepted / drafted - self.global_acceptance)

    def release(self, seq_id: str) -> None:
        self.counts.pop(seq_id, None)

    def _blend(self, old: float, new: float) -> float:
        return (1 - self.smoothing) * old + self.smoothing * new
