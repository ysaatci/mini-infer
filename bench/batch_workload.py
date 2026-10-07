import random
from dataclasses import dataclass

from bench.workload import make_prompt


@dataclass(frozen=True)
class BatchRequest:
    id: str
    prompt_ids: list[int]
    output_len: int
    arrival_s: float  # seconds after the benchmark starts


def make_requests(
    tokenizer,
    num_requests: int,
    rate: float | None,
    seed: int = 0,
    prompt_range: tuple[int, int] = (64, 1024),
    output_range: tuple[int, int] = (32, 512),
) -> list[BatchRequest]:
    """Mixed-length requests, identical for every engine given the same seed.

    rate None: all arrive at once (offline throughput). Otherwise a Poisson process at `rate` requests
    per second: random gaps between arrivals, as when independent users send requests.
    """
    rng = random.Random(seed)
    # Each prompt is a different slice of one long text, so no two requests share a prefix.
    text = make_prompt(tokenizer, prompt_range[1] * 4)[0].tolist()
    requests, clock = [], 0.0
    for i in range(num_requests):
        length = rng.randint(*prompt_range)
        start = rng.randint(0, len(text) - length)
        if rate is not None:
            clock += rng.expovariate(rate)
        requests.append(BatchRequest(str(i), text[start : start + length], rng.randint(*output_range), clock))
    return requests
