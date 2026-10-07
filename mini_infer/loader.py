from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from mini_infer.attention import AttentionBackend, SdpaBackend
from mini_infer.config import ModelConfig
from mini_infer.model import Qwen2ForCausalLM


# Config, tokenizer and weights. Everything else in a model repo (README, license) is skipped.
MODEL_FILES = ["*.json", "*.safetensors", "*.txt"]


def resolve_model_dir(name_or_path: str) -> Path:
    """Local directory as-is, otherwise a Hugging Face repo id already in the local cache."""
    path = Path(name_or_path)
    if path.is_dir():
        return path
    return Path(snapshot_download(name_or_path, allow_patterns=MODEL_FILES, local_files_only=True))


def load_model(
    name_or_path: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    backend: AttentionBackend | None = None,
) -> Qwen2ForCausalLM:
    model_dir = resolve_model_dir(name_or_path)
    config = ModelConfig.from_json(model_dir / "config.json")

    # Meta tensors have no storage, so we skip random init and never hold two copies of the weights.
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config, backend or SdpaBackend())

    state = {}
    for file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(file, framework="pt") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).to(device=device, dtype=dtype)

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    model.tie_weights()
    return model.eval()
