"""Batch-invariant mode: a request's output doesn't depend on which other requests share its batch.

Floating-point addition isn't associative, so a kernel whose summation order depends on the batch
(cuBLAS picks algorithms by row count; tuned Triton tiles depend on it too; attention chose its split
count from the batch size) gives a row slightly different numbers in different company, and greedy
decoding can then pick another token. Here every decode-path kernel sums each row in the same order
whatever the batch. Prefill needs no change: each prompt is prefilled alone.
"""

from torch import nn

from mini_infer.attention import TritonPagedBackend
from mini_infer.linear import LINEAR_LAYERS, TRITON_MAX_ROWS, TritonLinear
from mini_infer.model import Qwen2ForCausalLM
from mini_infer.quant import Int8Linear

SPLIT_BLOCKS = 32  # attention splits every 512 tokens


def make_batch_invariant(model: Qwen2ForCausalLM) -> Qwen2ForCausalLM:
    """Switch a loaded model (bf16 or int8) to batch-invariant kernels, in place."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LINEAR_LAYERS:
                if hasattr(parent, name):
                    setattr(parent, name, _invariant(getattr(parent, name)))
        backend = layer.self_attn.backend
        if not isinstance(backend, TritonPagedBackend):
            raise ValueError("batch-invariant mode needs the Triton attention backend")
        backend.split_blocks = SPLIT_BLOCKS
    model.lm_head = _invariant(model.lm_head)
    return model


def _invariant(layer: nn.Module) -> nn.Module:
    """Fixed-tile kernels for every decode-sized forward (prefills of one prompt don't mix requests)."""
    if isinstance(layer, Int8Linear):
        layer.invariant_max_rows = TRITON_MAX_ROWS
        return layer
    if not isinstance(layer, TritonLinear):
        layer = TritonLinear(layer)
    layer.invariant = True
    return layer
