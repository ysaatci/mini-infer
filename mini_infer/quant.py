"""int8 weight-only quantization: weights stored as int8 with one scale per output channel.

Decode reads every weight once per step and does little math with each, so it's limited by memory
bandwidth. Halving the bytes per weight should nearly halve the time spent reading them.
"""

import functools

import torch
import torch.nn.functional as F
import triton
import triton.testing
import triton.language as tl
from torch import Tensor, nn

from mini_infer.model import Qwen2ForCausalLM

# Up to this many rows (tokens in a forward) the Triton kernel runs: decode and speculative verification.
# Above it (prefill) the matmul is limited by compute, not memory, and cuBLAS on bf16 is faster.
KERNEL_MAX_ROWS = 64


def quantize_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """weight [N, K] -> (int8 [N, K], scale [N]) with weight ≈ int8 * scale[:, None].

    Symmetric, one scale per output row: the row's largest |w| maps to 127. Per-row scales keep a
    single large row from crushing the precision of every other row.
    """
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(weight.dtype)


def _configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=s)
        for n in (32, 64, 128)
        for k in (64, 128, 256)
        for w, s in ((4, 3),)
    ]


# Tuned once per (shape, row bucket) on first call. Short timing runs: ~45 tunings happen at startup.
@triton.autotune(
    configs=_configs(),
    key=["N", "K", "BLOCK_M"],
    do_bench=functools.partial(triton.testing.do_bench, warmup=5, rep=10),
)
@triton.jit
def _int8_matmul_kernel(
    X, W, Scale, Bias, Out, M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr,
):
    # out[M, N] = x[M, K] @ (w_int8[N, K] * scale[N])^T. Each program owns a BLOCK_M x BLOCK_N tile of the
    # output. Only int8 bytes come from memory; they become bf16 in registers, right before the tensor cores.
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        x = tl.load(X + rm[:, None] * stride_xm + k[None, :] * stride_xk, mask=(rm[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None] * stride_wn + k[None, :] * stride_wk, mask=(rn[:, None] < N) & (k[None, :] < K), other=0)
        acc += tl.dot(x, tl.trans(w.to(x.dtype)), input_precision=PRECISION)
    # The scale is per output column, so it factors out of the sum: apply it once, after accumulating.
    acc *= tl.load(Scale + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    if HAS_BIAS:
        acc += tl.load(Bias + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    tl.store(Out + rm[:, None] * stride_om + rn[None, :] * stride_on, acc.to(Out.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def int8_matmul(x: Tensor, weight: Tensor, scale: Tensor, bias: Tensor | None = None) -> Tensor:
    """x [M, K] @ (weight int8 [N, K] * scale [N])^T + bias -> [M, N], in x's dtype."""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    # tl.dot needs at least 16 rows. Larger tiles for larger M, so the weights are read once, not once per tile.
    block_m = min(64, max(16, triton.next_power_of_2(M)))
    grid = lambda meta: (triton.cdiv(M, block_m), triton.cdiv(N, meta["BLOCK_N"]))
    _int8_matmul_kernel[grid](
        x, weight, scale, bias if bias is not None else scale, out, M, N, K,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        PRECISION="ieee" if x.dtype == torch.float32 else "tf32",
    )
    return out


class Int8Linear(nn.Module):
    """Drop-in for nn.Linear with int8 weights and per-output-channel scales."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        q, scale = quantize_weight(linear.weight.data)
        self.register_buffer("weight", q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", linear.bias.data if linear.bias is not None else None)

    def forward(self, x: Tensor) -> Tensor:
        rows = x.reshape(-1, x.shape[-1])
        if rows.shape[0] <= KERNEL_MAX_ROWS:
            out = int8_matmul(rows, self.weight, self.scale, self.bias)
        else:
            # Prefill: rebuild this one layer's weights in bf16 for cuBLAS. Freed right after, so memory
            # never holds more than one layer's dequantized copy.
            out = F.linear(rows, self.weight.to(x.dtype) * self.scale[:, None].to(x.dtype), self.bias)
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
