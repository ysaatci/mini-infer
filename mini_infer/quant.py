"""int8 weight-only quantization: weights stored as int8 with one scale per output channel.

Decode reads every weight once per step and does little math with each, so it's limited by memory
bandwidth. Halving the bytes per weight should nearly halve the time spent reading them.
"""

import torch
from torch import Tensor, nn

from mini_infer.matmul import matmul
from mini_infer.model import Qwen2ForCausalLM


def quantize_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """weight [N, K] -> (int8 [N, K], scale [N]) with weight ≈ int8 * scale[:, None].

    Symmetric, one scale per output row: the row's largest |w| maps to 127. Per-row scales keep a
    single large row from crushing the precision of every other row.
    """
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(weight.dtype)


def int8_matmul(x: Tensor, weight: Tensor, scale: Tensor, bias: Tensor | None = None) -> Tensor:
    """x [M, K] @ (weight int8 [N, K] * scale [N])^T + bias -> [M, N], in x's dtype."""
    return matmul(x, weight, scale, bias)


class Int8Linear(nn.Module):
    """Drop-in for nn.Linear with int8 weights and per-output-channel scales."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        q, scale = quantize_weight(linear.weight.data)
        self.register_buffer("weight", q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", linear.bias.data if linear.bias is not None else None)
        self.invariant_max_rows = 0  # deterministic mode: forwards up to this many rows use fixed tiles

    def forward(self, x: Tensor) -> Tensor:
        rows = x.reshape(-1, x.shape[-1])
        out = matmul(rows, self.weight, self.scale, self.bias, invariant=rows.shape[0] <= self.invariant_max_rows)
        return out.view(*x.shape[:-1], -1)


QUANTIZED_LAYERS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@torch.no_grad()
def quantize_model(model: Qwen2ForCausalLM) -> Qwen2ForCausalLM:
    """Replace every decoder linear layer and the output head with Int8Linear, in place.

    One layer at a time, so peak memory grows by one layer's int8 copy, not a whole second model.
    The output head shares its matrix with the input embedding; it gets its own int8 copy while the
    embedding stays bf16, since a lookup only reads the rows of the tokens in the batch.
    """
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in QUANTIZED_LAYERS:
                if hasattr(parent, name):
                    setattr(parent, name, Int8Linear(getattr(parent, name)))
    model.lm_head = Int8Linear(model.lm_head)
    torch.cuda.empty_cache()
    return model
