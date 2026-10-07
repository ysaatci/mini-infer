#!/usr/bin/env bash
# Runs a Python script or pytest inside the WSL venv: bash scripts/wsl_run.sh <args...>
set -euo pipefail
export PATH="/root/venvs/mini-infer/bin:/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.."
exec "$@"
