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


@dataclass(frozen=True)
class BatchMetrics:
    output_tok_s: float  # generated tokens per second across all requests: the throughput headline
    requests_s: float
    ttft_p50_ms: float  # from a request's arrival to its first token, includes queueing
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p99_ms: float  # spikes here are running requests paused by other requests' prefills
    e2e_p50_s: float
    e2e_p99_s: float
    peak_mem_gb: float


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


def summarize_batch(
    token_times: dict[str, list[float]], arrivals: dict[str, float], makespan_s: float, peak_mem_bytes: int
) -> BatchMetrics:
    """token_times: per request, when each token arrived (seconds since benchmark start)."""
    ttfts = [times[0] - arrivals[rid] for rid, times in token_times.items()]
    e2es = [times[-1] - arrivals[rid] for rid, times in token_times.items()]
    itls = [b - a for times in token_times.values() for a, b in zip(times, times[1:])]
    return BatchMetrics(
        output_tok_s=sum(len(t) for t in token_times.values()) / makespan_s,
        requests_s=len(token_times) / makespan_s,
        ttft_p50_ms=percentile(ttfts, 50) * 1e3,
        ttft_p99_ms=percentile(ttfts, 99) * 1e3,
        itl_p50_ms=percentile(itls, 50) * 1e3,
        itl_p99_ms=percentile(itls, 99) * 1e3,
        e2e_p50_s=percentile(e2es, 50),
        e2e_p99_s=percentile(e2es, 99),
        peak_mem_gb=peak_mem_bytes / 1e9,
    )
