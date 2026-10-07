#!/usr/bin/env bash
# Creates the venv outside the repo (Windows mounts are slow) and checks the GPU.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export UV_PROJECT_ENVIRONMENT="$HOME/venvs/mini-infer"
export UV_LINK_MODE=copy
cd "$(dirname "$0")/.."
uv sync
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/check_gpu.py
