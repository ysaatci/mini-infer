import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    ttft_ms: float  # time to first token: prefill + first sample
    decode_tok_s: float  # tokens per second after the first
    itl_p50_ms: float  # inter-token latency, median
    itl_p99_ms: float  # inter-token latency, worst 1%
    e2e_s: float  # whole request
    peak_mem_gb: float  # weights + cache + activations


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def summarize(runs: list[list[float]], peak_mem_bytes: int) -> Metrics:
    """runs: per repeat, the time each token arrived. Medians across repeats damp one-off noise."""
    itls = [b - a for times in runs for a, b in zip(times, times[1:])]
    return Metrics(
        ttft_ms=statistics.median(t[0] for t in runs) * 1e3,
        decode_tok_s=statistics.median((len(t) - 1) / (t[-1] - t[0]) for t in runs),
        itl_p50_ms=percentile(itls, 50) * 1e3,
        itl_p99_ms=percentile(itls, 99) * 1e3,
        e2e_s=statistics.median(t[-1] for t in runs),
        peak_mem_gb=peak_mem_bytes / 1e9,
    )
