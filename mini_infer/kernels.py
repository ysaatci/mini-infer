import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _paged_decode_kernel(
    Q, K, V, Out, BlockTables, SeqLens, scale,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kt,
    stride_tb,
    stride_ob, stride_oh,
    GROUP: tl.constexpr,  # query heads per k/v head
    GROUP_PAD: tl.constexpr,  # tl.dot needs at least 16 rows
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PRECISION: tl.constexpr,
):
    # One program per (sequence, k/v head). It serves all query heads sharing that k/v head,
    # so each block of keys/values is loaded once and used GROUP times.
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq_len = tl.load(SeqLens + b)

    g = tl.arange(0, GROUP_PAD)
    d = tl.arange(0, HEAD_DIM)
    t = tl.arange(0, BLOCK_SIZE)
    heads = kv_head * GROUP + g
    head_mask = g < GROUP
    q = tl.load(Q + b * stride_qb + heads[:, None] * stride_qh + d[None, :], mask=head_mask[:, None], other=0.0)

    # Online softmax: running max, running sum and weighted value total, updated block by block,
    # so the full row of attention scores never has to exist in memory.
    m = tl.full([GROUP_PAD], float("-inf"), tl.float32)
    s_sum = tl.zeros([GROUP_PAD], tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], tl.float32)

    for j in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        block = tl.load(BlockTables + b * stride_tb + j).to(tl.int64)
        offsets = block * stride_kb + kv_head * stride_kh + t[:, None] * stride_kt + d[None, :]
        k = tl.load(K + offsets)  # [BLOCK_SIZE, HEAD_DIM], read in place: no gather
        v = tl.load(V + offsets)

        scores = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale  # [GROUP_PAD, BLOCK_SIZE]
        valid = j * BLOCK_SIZE + t < seq_len  # the last block is only partly filled
        scores = tl.where(valid[None, :], scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, 1))
        rescale = tl.exp(m - m_new)  # earlier totals were computed against a smaller max
        p = tl.exp(scores - m_new[:, None])
        s_sum = s_sum * rescale + tl.sum(p, 1)
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
        m = m_new

    out = acc / s_sum[:, None]
    tl.store(Out + b * stride_ob + heads[:, None] * stride_oh + d[None, :], out.to(Out.dtype.element_ty), mask=head_mask[:, None])


def paged_decode_attention(q: Tensor, k_cache: Tensor, v_cache: Tensor, block_tables: Tensor, seq_lens: Tensor) -> Tensor:
    """One new token per sequence.

    q [B, heads, 1, D]; k/v_cache [num_blocks, kv_heads, block_size, D] for one layer;
    block_tables [B, max_blocks] int32; seq_lens [B] tokens per sequence including the new one.
    Returns [B, heads, 1, D].
    """
    B, H, _, D = q.shape
    _, H_kv, block_size, _ = k_cache.shape
    group = H // H_kv
    q = q.reshape(B, H, D).contiguous()
    out = torch.empty_like(q)
    _paged_decode_kernel[(B, H_kv)](
        q, k_cache, v_cache, out, block_tables, seq_lens, D**-0.5,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        GROUP=group,
        GROUP_PAD=max(16, triton.next_power_of_2(group)),
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        # fp32 inputs would otherwise be rounded to tf32 inside tl.dot
        PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
    )
    return out.view(B, H, 1, D)
