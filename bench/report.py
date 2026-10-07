import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from bench.metrics import Metrics

Row = tuple[str, str, Metrics]  # engine, workload, metrics


def environment() -> dict:
    """What a reader needs to reproduce or compare a run."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }


def save_json(path: Path, settings: dict, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "environment": environment(),
        "settings": settings,
        "results": [{"engine": e, "workload": w, **asdict(m)} for e, w, m in rows],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def markdown_table(rows: list[Row]) -> str:
    lines = [
        "| Engine | Workload | TTFT ms | Decode tok/s | ITL p50 ms | ITL p99 ms | Total s | Peak GB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for engine, workload, m in rows:
        lines.append(
            f"| {engine} | {workload} | {m.ttft_ms:.1f} | {m.decode_tok_s:.1f} | {m.itl_p50_ms:.1f} "
            f"| {m.itl_p99_ms:.1f} | {m.e2e_s:.2f} | {m.peak_mem_gb:.2f} |"
        )
    return "\n".join(lines)
