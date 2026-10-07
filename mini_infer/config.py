import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        raw = json.loads(Path(path).read_text())
        if raw["model_type"] != "qwen2":
            raise ValueError(f"unsupported model_type: {raw['model_type']}")
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_layers=raw["num_hidden_layers"],
            num_heads=raw["num_attention_heads"],
            num_kv_heads=raw["num_key_value_heads"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            tie_word_embeddings=raw["tie_word_embeddings"],
        )
