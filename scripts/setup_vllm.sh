#!/usr/bin/env bash
# vLLM baseline in its own venv: it pins its own torch version, which must not leak into ours.
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
uv venv "$HOME/venvs/vllm" --python 3.12 --allow-existing
uv pip install --python "$HOME/venvs/vllm/bin/python" vllm
"$HOME/venvs/vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda)"
