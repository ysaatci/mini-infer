"""Batch-invariant mode: a request's output doesn't depend on which other requests share its batch.

Floating-point addition isn't associative, so a kernel whose summation order depends on the batch
(cuBLAS picks algorithms by row count; attention chose its split count from the batch size) gives a
row slightly different numbers in different company, and greedy decoding can then pick another token.
Here every decode-path kernel sums each row in the same order whatever the batch. Prefill needs no
change: each prompt is prefilled alone.
"""

import torch.nn.functional as F
from torch import Tensor, nn

from mini_infer.attention import TritonPagedBackend
from mini_infer.matmul import matmul
from mini_infer.model import Qwen2ForCausalLM
from mini_infer.quant import QUANTIZED_LAYERS, Int8Linear

# Decode (up to 64 requests) and speculative verification (up to 16 requests x 5 tokens) stay below this.
# Larger forwards are prefills of a single prompt, which don't mix requests, so they keep cuBLAS.
INVARIANT_MAX_ROWS = 128
SPLIT_BLOCKS = 32  # attention splits every 512 tokens


class InvariantLinear(nn.Module):
    """nn.Linear with bf16 weights whose decode-sized forwards use fixed-tile Triton instead of cuBLAS."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x: Tensor) -> Tensor:
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[0] > INVARIANT_MAX_ROWS:
            return F.linear(x, self.weight, self.bias)
        return matmul(rows, self.weight, bias=self.bias, invariant=True).view(*x.shape[:-1], -1)


def make_batch_invariant(model: Qwen2ForCausalLM) -> Qwen2ForCausalLM:
    """Switch a loaded model (bf16 or int8) to batch-invariant kernels, in place."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in QUANTIZED_LAYERS:
                if hasattr(parent, name):
                    setattr(parent, name, _invariant(getattr(parent, name)))
        backend = layer.self_attn.backend
        if not isinstance(backend, TritonPagedBackend):
            raise ValueError("batch-invariant mode needs the Triton attention backend")
        backend.split_blocks = SPLIT_BLOCKS
    model.lm_head = _invariant(model.lm_head)
    return model


def _invariant(layer: nn.Module) -> nn.Module:
    if isinstance(layer, Int8Linear):
        layer.invariant_max_rows = INVARIANT_MAX_ROWS
        return layer
    return InvariantLinear(layer)
