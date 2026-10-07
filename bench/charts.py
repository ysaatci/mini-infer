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

# Validated palette (two categorical slots, both modes) plus gray for reference systems (HF, vLLM).
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834"], "reference": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926"], "reference": "#898781"},
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


def speculative(theme: dict):
    """Change in output speed from speculative decoding (k = 2) by how many requests are in flight."""
    concurrency = [1, 4, 8]
    fig, ax = figure(theme, 7.2, 3.2)
    width = 0.34
    for i, (name, temperature) in enumerate((("greedy", "0"), ("temperature 0.7", "0.7"))):
        gains = []
        for c in concurrency:
            base = metric("step7-spec.json", "no-spec", f"c{c}-t{temperature}", "output_tok_s")
            spec = metric("step7-spec.json", "k=2", f"c{c}-t{temperature}", "output_tok_s")
            gains.append((spec / base - 1) * 100)
        xs = [x + (i - 0.5) * (width + 0.04) for x in range(len(concurrency))]
        ax.bar(xs, gains, width=width, color=theme["series"][i], label=name)
        for x, g in zip(xs, gains):
            ax.text(x, g + (1.5 if g >= 0 else -1.5), f"{g:+.0f}%", ha="center", va="bottom" if g >= 0 else "top",
                    color=theme["ink"], fontsize=9)
    style(ax, theme, "y")
    ax.axhline(0, color=theme["axis"], linewidth=0.8)
    ax.set_xticks(range(len(concurrency)), [f"{c} in flight" for c in concurrency])
    ax.set_ylabel("change in output tokens/s, %", color=theme["muted"], fontsize=9)
    ax.set_ylim(-35, 55)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    title(ax, "Speculative decoding: 0.5B drafts 2 tokens, 1.5B verifies", theme)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller files, selectable
    for name, chart in (("single-request", single_request), ("batching", batching), ("speculative", speculative)):
        for mode, theme in THEMES.items():
            fig = chart(theme)
            fig.savefig(OUT / f"{name}-{mode}.svg", facecolor=theme["surface"], bbox_inches="tight")
            plt.close(fig)
            print(OUT / f"{name}-{mode}.svg")


if __name__ == "__main__":
    main()
