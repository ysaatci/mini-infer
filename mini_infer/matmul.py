"""Triton matmul for linear layers: x @ (w * scale)^T + bias, with w in bf16 or int8.

Two ways to run it:
- tuned: tile sizes picked per (shape, row-count bucket) on first use. Fastest, but a row's summation
  order can depend on how many rows share the call.
- invariant: one fixed tile configuration for every row count. Each row is always summed in the same
  order, so its result doesn't depend on what else is in the batch (deterministic mode).
"""

import functools

import torch
import triton
import triton.language as tl
import triton.testing
from torch import Tensor

# Up to this many rows (tokens in a forward) the matmul is memory-bound: decode and speculative
# verification. Above it (prefill) it's compute-bound and wants larger tiles.
DECODE_MAX_ROWS = 64


@triton.jit
def _matmul_kernel(
    X, W, Scale, Bias, Out, M, N, K, M_BUCKET,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
    HAS_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr,
):
    # Each program owns a BLOCK_M x BLOCK_N tile of the output. int8 weights are converted to the
    # activation dtype in registers, right before the tensor cores, so only int8 bytes come from memory.
    # M_BUCKET is only an autotuning key (best tiles depend on roughly how many rows there are).
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        x = tl.load(X + rm[:, None] * stride_xm + k[None, :] * stride_xk, mask=(rm[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None] * stride_wn + k[None, :] * stride_wk, mask=(rn[:, None] < N) & (k[None, :] < K), other=0)
        acc += tl.dot(x, tl.trans(w.to(x.dtype)), input_precision=PRECISION)
    if HAS_SCALE:
        # Per output column, so it factors out of the sum: applied once, after accumulating.
        acc *= tl.load(Scale + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    if HAS_BIAS:
        acc += tl.load(Bias + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    tl.store(Out + rm[:, None] * stride_om + rn[None, :] * stride_on, acc.to(Out.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


# Two tuners around the same kernel. Short timing runs keep startup tuning to a few seconds.
_bench = functools.partial(triton.testing.do_bench, warmup=5, rep=10)

# Decode: few rows, memory-bound. The row tile is fixed by the caller (just enough rows), and the tuner
# picks how wide a slice of the weights each program streams through.
_decode_matmul = triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": n, "BLOCK_K": k}, num_warps=4, num_stages=3) for n in (32, 64, 128) for k in (64, 128, 256)
    ],
    key=["N", "K", "BLOCK_M"],
    do_bench=_bench,
)(_matmul_kernel)

# Prefill: many rows, compute-bound, so larger tiles that reuse each loaded weight across more rows.
# The configs that won a search over 32 candidates at 128-2048 rows (within ~10% of bf16 cuBLAS).
_prefill_matmul = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k}, num_warps=w, num_stages=s)
        for m, n, k, w, s in (
            (64, 64, 32, 4, 4), (64, 128, 32, 4, 4), (64, 64, 64, 8, 3),
            (64, 128, 64, 4, 3), (64, 128, 64, 4, 4), (128, 128, 32, 8, 4),
        )
    ],
    key=["N", "K", "M_BUCKET"],
    do_bench=_bench,
)(_matmul_kernel)

# Invariant: the same tiles whatever the row count. 16 rows per program (the tl.dot minimum), so a row's
# result never depends on which rows share its tile.
INVARIANT_CONFIG = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}


def matmul(x: Tensor, weight: Tensor, scale: Tensor | None = None, bias: Tensor | None = None, invariant: bool = False) -> Tensor:
    """x [M, K] @ (weight [N, K] * scale [N])^T + bias -> [M, N], in x's dtype. weight is bf16 or int8."""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    placeholder = weight  # pointer for absent scale/bias; never read (HAS_SCALE/HAS_BIAS are False)
    args = (
        x, weight, scale if scale is not None else placeholder, bias if bias is not None else placeholder, out, M, N, K,
        _row_bucket(M),
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), out.stride(0), out.stride(1),
    )
    flags = {
        "HAS_SCALE": scale is not None,
        "HAS_BIAS": bias is not None,
        "PRECISION": "ieee" if x.dtype == torch.float32 else "tf32",
    }
    if invariant:
        c = INVARIANT_CONFIG
        grid = (triton.cdiv(M, c["BLOCK_M"]), triton.cdiv(N, c["BLOCK_N"]))
        _matmul_kernel[grid](*args, **flags, **c)
    elif M <= DECODE_MAX_ROWS:
        # tl.dot needs at least 16 rows. Larger tiles for larger M, so the weights are read once, not once per tile.
        block_m = min(64, max(16, triton.next_power_of_2(M)))
        grid = lambda meta: (triton.cdiv(M, block_m), triton.cdiv(N, meta["BLOCK_N"]))
        _decode_matmul[grid](*args, BLOCK_M=block_m, **flags)
    else:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        _prefill_matmul[grid](*args, **flags)
    return out


def _row_bucket(rows: int) -> int:
    """Coarse prompt-size class for prefill tuning: re-tuning for every prompt length would be slow."""
    return 256 if rows <= 256 else 1024 if rows <= 1024 else 4096
