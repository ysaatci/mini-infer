"""README charts, generated from bench/results/*.json. Each chart is written in a light and a dark variant.

python -m bench.charts
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parent.parent / "docs" / "charts"

# Validated palette (three categorical slots, both modes) plus gray for reference systems (HF, vLLM).
# The light third slot is below 3:1 contrast, so every line using it also gets a direct label.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"], "reference": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70"], "reference": "#898781"},
}


def metric(file: str, engine: str, workload: str, field: str) -> float:
    data = json.loads((RESULTS / file).read_text())
    return next(r[field] for r in data["results"] if r["engine"] == engine and r["workload"] == workload)


def style(ax, theme: dict, grid_axis: str) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(theme: dict, width: float, height: float, columns: int = 1):
    fig, axes = plt.subplots(1, columns, figsize=(width, height))
    fig.patch.set_facecolor(theme["surface"])
    return fig, axes


def title(ax, text: str, theme: dict) -> None:
    ax.set_title(text, loc="left", color=theme["ink"], fontsize=11, pad=10)


def single_request(theme: dict):
    """Decode speed of one request after each optimization, against Hugging Face."""
    rows = [
        ("Hugging Face generate", metric("step3-baseline.json", "hf", "in128-out256", "decode_tok_s"), True),
        ("No KV cache", metric("step3-baseline.json", "mini-infer-nocache", "in128-out256", "decode_tok_s"), False),
        ("KV cache", metric("step3-baseline.json", "mini-infer", "in128-out256", "decode_tok_s"), False),
        ("+ paged cache, Triton kernel", metric("step5-single.json", "mini-infer", "in128-out256", "decode_tok_s"), False),
        ("+ CUDA graphs", metric("graphs-single.json", "mini-infer", "in128-out256", "decode_tok_s"), False),
        ("+ int8 weights", metric("step8b-single.json", "mini-infer-int8", "in128-out256", "decode_tok_s"), False),
    ]
    fig, ax = figure(theme, 7.2, 3.2)
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = [theme["reference"] if r[2] else theme["series"][0] for r in rows][::-1]
    ax.barh(labels, values, color=colors, height=0.55)
    for y, v in enumerate(values):
        ax.text(v + 2, y, f"{v:.0f}", va="center", color=theme["ink"], fontsize=9)
    style(ax, theme, "x")
    ax.tick_params(axis="y", colors=theme["ink"])
    ax.set_xlabel("decode tokens/s, one request (higher is better)", color=theme["muted"], fontsize=9)
    ax.set_xlim(0, max(values) * 1.12)
    title(ax, "Single request: Qwen2.5-1.5B, 128-token prompt", theme)
    return fig


def batching(theme: dict):
    """Throughput with everything submitted at once, and request latency as load grows, against vLLM."""
    fig, (left, right) = figure(theme, 9.6, 3.4, columns=2)

    rows = [
        ("vLLM 0.31", metric("step5-vllm-offline.json", "vllm", "offline-n200", "output_tok_s"), True),
        ("Continuous batching", metric("step4-mini-infer-offline.json", "mini-infer", "offline-n200", "output_tok_s"), False),
        ("+ paged cache, kernel", metric("step5-mini-infer-offline.json", "mini-infer", "offline-n200", "output_tok_s"), False),
        ("+ CUDA graphs", metric("graphs-mini-infer-offline.json", "mini-infer", "offline-n200", "output_tok_s"), False),
        ("+ int8 weights", metric("step8b-int8-offline.json", "mini-infer-int8", "offline-n200", "output_tok_s"), False),
    ]
    labels, values = [r[0] for r in rows][::-1], [r[1] for r in rows][::-1]
    colors = [theme["reference"] if r[2] else theme["series"][0] for r in rows][::-1]
    left.barh(labels, values, color=colors, height=0.55)
    for y, v in enumerate(values):
        left.text(v + 15, y, f"{v:.0f}", va="center", color=theme["ink"], fontsize=9)
    style(left, theme, "x")
    left.tick_params(axis="y", colors=theme["ink"])
    left.set_xlim(0, max(values) * 1.15)
    left.set_xlabel("output tokens/s, 200 requests at once", color=theme["muted"], fontsize=9)
    title(left, "Throughput", theme)

    rates = [1, 2, 3]
    series = [
        ("mini-infer bf16", [metric(f"graphs-mini-infer-rate{r}.json", "mini-infer", f"rate{r}-n100", "e2e_p50_s") for r in rates], theme["series"][0]),
        ("mini-infer int8", [metric(f"step8b-int8-rate{r}.json", "mini-infer-int8", f"rate{r}-n100", "e2e_p50_s") for r in rates], theme["series"][1]),
        ("vLLM 0.31 bf16", [metric(f"step5-vllm-rate{r}.json", "vllm", f"rate{r}-n100", "e2e_p50_s") for r in rates], theme["reference"]),
    ]
    for name, values, color in series:
        right.plot(rates, values, color=color, linewidth=2, marker="o", markersize=7,
                   markeredgecolor=theme["surface"], markeredgewidth=2, label=name)
    style(right, theme, "y")
    right.set_xticks(rates, [f"{r} req/s" for r in rates])
    right.set_ylim(0, None)
    right.set_ylabel("median request time, s (lower is better)", color=theme["muted"], fontsize=9)
    legend = right.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    title(right, "Latency as load grows", theme)
    fig.tight_layout(w_pad=3)
    return fig


def speculation_policies(theme: dict):
    """Each speculation policy's throughput against plain decode, from 1 to 16 requests in flight."""
    data = json.loads((RESULTS / "step10-spec.json").read_text())["results"]
    tok_s = {(r["engine"], r["workload"]): r["output_tok_s"] for r in data}
    concurrency = [1, 2, 4, 8, 16]
    series = [("adaptive", "adaptive", 0), ("fixed-2", "fixed k = 2", 1), ("fixed-4", "fixed k = 4", 2)]
    fig, axes = figure(theme, 9.6, 3.4, columns=2)
    for ax, temperature, name in zip(axes, ("0", "0.7"), ("greedy", "temperature 0.7")):
        for policy, label, slot in series:
            change = [(tok_s[(policy, f"c{c}-t{temperature}")] / tok_s[("none", f"c{c}-t{temperature}")] - 1) * 100
                      for c in concurrency]
            x = range(len(concurrency))
            ax.plot(x, change, color=theme["series"][slot], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=theme["surface"], markeredgewidth=2, label=label)
            ax.text(len(concurrency) - 0.85, change[-1], label, va="center", color=theme["ink"], fontsize=8.5)
        ax.axhline(0, color=theme["reference"], linewidth=1)
        style(ax, theme, "y")
        ax.set_xticks(range(len(concurrency)), [str(c) for c in concurrency])
        ax.set_xlim(-0.3, len(concurrency) + 0.6)
        ax.set_ylim(-75, 35)
        ax.set_xlabel("requests in flight", color=theme["muted"], fontsize=9)
        title(ax, name, theme)
    axes[0].set_ylabel("output tokens/s vs plain decode, %", color=theme["muted"], fontsize=9)
    fig.tight_layout(w_pad=3)
    return fig


def adaptive_trace(theme: dict):
    """Requests in flight and the k the adaptive policy chose, while load goes quiet, busy, quiet."""
    data = json.loads((RESULTS / "step10-trace.json").read_text())
    steps = data["stats"]["adaptive_trace"]  # (seconds, batch size, k) per decode step
    seconds = range(int(steps[-1][0]) + 1)
    batch, k = [], []
    for second in seconds:  # 1-second averages: thousands of steps are unreadable as points
        window = [s for s in steps if second <= s[0] < second + 1]
        batch.append(sum(s[1] for s in window) / len(window) if window else 0)
        k.append(sum(s[2] for s in window) / len(window) if window else float("nan"))  # idle: no decision, a gap

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.6, 3.8), sharex=True, height_ratios=[1, 1])
    fig.patch.set_facecolor(theme["surface"])
    busy_start, busy_end = data["settings"]["phases"][0][0], sum(p[0] for p in data["settings"]["phases"][:2])
    for ax, values, label, color in ((top, batch, "requests in flight", theme["reference"]),
                                     (bottom, k, "drafted tokens per step (k)", theme["series"][0])):
        ax.axvspan(busy_start, busy_end, color=theme["grid"], linewidth=0)
        ax.plot(list(seconds), values, color=color, linewidth=2)
        style(ax, theme, "y")
        ax.set_ylabel(label, color=theme["muted"], fontsize=9)
    top.text(busy_start + 1, max(batch) * 0.9, "busy: 4 requests/s", color=theme["muted"], fontsize=8.5)
    bottom.set_ylim(0, 4.2)
    bottom.set_xlabel("seconds", color=theme["muted"], fontsize=9)
    title(top, "Adaptive speculation: drafts while quiet, backs off while busy", theme)
    fig.tight_layout(h_pad=1)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller files, selectable
    charts = (
        ("single-request", single_request),
        ("batching", batching),
        ("speculation-policies", speculation_policies),
        ("adaptive-trace", adaptive_trace),
    )
    for name, chart in charts:
        for mode, theme in THEMES.items():
            fig = chart(theme)
            fig.savefig(OUT / f"{name}-{mode}.svg", facecolor=theme["surface"], bbox_inches="tight")
            plt.close(fig)
            print(OUT / f"{name}-{mode}.svg")


if __name__ == "__main__":
    main()
