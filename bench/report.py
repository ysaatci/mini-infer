import json
import subprocess
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

import torch

Row = tuple[str, str, object]  # engine, workload, a metrics dataclass


def environment() -> dict:
    """What a reader needs to reproduce or compare a run."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }


def save_json(path: Path, settings: dict, rows: list[Row], stats: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "environment": environment(),
        "settings": settings,
        "results": [{"engine": e, "workload": w, **asdict(m)} for e, w, m in rows],
        "stats": stats or {},
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def markdown_table(rows: list[Row]) -> str:
    """One column per metrics field, so any metrics dataclass prints the same way."""
    names = [f.name for f in fields(rows[0][2])]
    lines = ["| engine | workload | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    for engine, workload, m in rows:
        values = " | ".join(f"{getattr(m, n):.2f}" for n in names)
        lines.append(f"| {engine} | {workload} | {values} |")
    return "\n".join(lines)
