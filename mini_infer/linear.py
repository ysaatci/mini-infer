"""Linear layers backed by the Triton matmul for decode-sized forwards.

cuBLAS is slow at decode sizes on this GPU: a 1-row 1536 x 1536 bf16 matmul took 36.5 us against 15.6 us
for the Triton kernel, and a whole decode step ran ~35% faster with it. Prefill-sized forwards (one
prompt, hundreds of rows) are compute-bound, where cuBLAS is at its best, so they keep it.
"""

import torch.nn.functional as F
from torch import Tensor, nn

from mini_infer.matmul import matmul

# Decode (up to 64 requests) and speculative verification (up to 16 requests x 5 tokens) stay below this.
TRITON_MAX_ROWS = 128
LINEAR_LAYERS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class TritonLinear(nn.Module):
    """nn.Linear whose forwards of up to TRITON_MAX_ROWS rows use the Triton matmul.

    invariant: fixed tiles instead of tuned ones, so a row's result never depends on the batch
    (deterministic mode). Holds the original weight and bias, so tied weights stay tied.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.invariant = False

    def forward(self, x: Tensor) -> Tensor:
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[0] > TRITON_MAX_ROWS:
            return F.linear(x, self.weight, self.bias)
        return matmul(rows, self.weight, bias=self.bias, invariant=self.invariant).view(*x.shape[:-1], -1)


def use_triton_linears(model) -> None:
    """Swap every decoder linear layer and the output head for TritonLinear, in place."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LINEAR_LAYERS:
                if hasattr(parent, name):
                    setattr(parent, name, TritonLinear(getattr(parent, name)))
    model.lm_head = TritonLinear(model.lm_head)
