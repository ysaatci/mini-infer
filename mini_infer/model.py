import torch
from torch import Tensor, nn

from mini_infer.attention import AttentionBackend
from mini_infer.cache import KVCache
from mini_infer.config import ModelConfig
from mini_infer.layers import MLP, RMSNorm, RotaryEmbedding, apply_rope

# Module names mirror Hugging Face's Qwen2 so checkpoint keys load without renaming.


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, backend: AttentionBackend, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim
        # Qwen2 uses bias on q/k/v but not on the output projection.
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.backend = backend

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None) -> Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = None
        if cache is not None:
            # Keys are cached after RoPE, so past tokens never need re-rotating.
            k, v, mask = cache.update(self.layer_idx, k, v)
        out = self.backend(q, k, v, mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, backend: AttentionBackend, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, backend, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config, backend, i) for i in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, input_ids: Tensor, positions: Tensor, cache: KVCache | None) -> Tensor:
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, cache)
        return self.norm(x)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config, backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def tie_weights(self) -> None:
        """Share the embedding matrix with the output head, as the checkpoint expects."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: Tensor, positions: Tensor | None = None, cache: KVCache | None = None) -> Tensor:
        """input_ids [B, T] -> logits [B, T, vocab]. With a cache, input_ids are only the new tokens."""
        T = input_ids.shape[1]
        if positions is None:
            start = cache.length if cache is not None else 0
            positions = torch.arange(start, start + T, device=input_ids.device).expand_as(input_ids)
        hidden = self.model(input_ids, positions, cache)
        if cache is not None:
            cache.advance(T)
        return self.lm_head(hidden)
