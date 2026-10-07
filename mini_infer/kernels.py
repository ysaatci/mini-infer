import functools

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _paged_attention_kernel(
    Q, K, V, Out, Maxes, Sums, BlockTables, ContextLens, scale, num_splits,
    stride_qb, stride_qh, stride_qt,
    stride_kb, stride_kh, stride_kt,
    stride_tb,
    stride_ob, stride_oh, stride_ot, stride_os,
    stride_mb, stride_mh, stride_mt,
    GROUP: tl.constexpr,  # query heads per k/v head
    QUERY_LEN: tl.constexpr,  # new tokens per sequence: 1 for decode, k + 1 for speculative verification
    ROWS_PAD: tl.constexpr,  # QUERY_LEN * GROUP rounded up; tl.dot needs at least 16 rows
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PRECISION: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # One program per (sequence, k/v head, split). Its rows are every (new token, query head) pair that
    # reads this k/v head, so each block of keys/values is loaded once and used QUERY_LEN * GROUP times.
    # Each split covers a range of the sequence's blocks, so a small batch still keeps the GPU busy.
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    context = tl.load(ContextLens + b)  # tokens cached before the new ones
    seq_len = context + QUERY_LEN

    r = tl.arange(0, ROWS_PAD)
    token = r // GROUP  # which new token this row is
    heads = kv_head * GROUP + r % GROUP
    row_mask = r < QUERY_LEN * GROUP
    d = tl.arange(0, HEAD_DIM)
    t = tl.arange(0, BLOCK_SIZE)
    q_offsets = b * stride_qb + heads[:, None] * stride_qh + token[:, None] * stride_qt + d[None, :]
    q = tl.load(Q + q_offsets, mask=row_mask[:, None], other=0.0)

    # Online softmax: running max, running sum and weighted value total, updated block by block,
    # so the full row of attention scores never has to exist in memory.
    m = tl.full([ROWS_PAD], float("-inf"), tl.float32)
    s_sum = tl.zeros([ROWS_PAD], tl.float32)
    acc = tl.zeros([ROWS_PAD, HEAD_DIM], tl.float32)

    # Each sequence is divided by its own length, so a short sequence next to a long one isn't left
    # with empty splits while the long one's splits do all the work.
    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    blocks_per_split = tl.cdiv(num_blocks, num_splits)
    start = split * blocks_per_split
    end = tl.minimum(start + blocks_per_split, num_blocks)
    for j in range(start, end):
        block = tl.load(BlockTables + b * stride_tb + j).to(tl.int64)
        offsets = block * stride_kb + kv_head * stride_kh + t[:, None] * stride_kt + d[None, :]
        k = tl.load(K + offsets)  # [BLOCK_SIZE, HEAD_DIM], read in place: no gather
        v = tl.load(V + offsets)

        scores = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale  # [ROWS_PAD, BLOCK_SIZE]
        # New token i sits at position context + i and sees keys up to there (causal among new tokens).
        # For a single new token this is just "the last block is only partly filled".
        visible = (j * BLOCK_SIZE + t)[None, :] <= (context + token)[:, None]
        scores = tl.where(visible, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, 1))
        # A row can have no visible key in this split yet (an early new token, a late split). Its max is
        # still -inf, and -inf - -inf is NaN, so subtract 0 instead: its terms are all exp(-inf) = 0.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        rescale = tl.exp(m - m_safe)  # earlier totals were computed against a smaller max
        p = tl.exp(scores - m_safe[:, None])
        s_sum = s_sum * rescale + tl.sum(p, 1)
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
        m = m_new

    out_offsets = b * stride_ob + heads[:, None] * stride_oh + token[:, None] * stride_ot + split * stride_os + d[None, :]
    if SPLIT:
        # Partial result for this range, merged across splits afterwards. A row that saw nothing here
        # stores max -inf and sum 0, so it gets zero weight in the merge.
        tl.store(Out + out_offsets, acc, mask=row_mask[:, None])
        stat_offsets = b * stride_mb + heads * stride_mh + token * stride_mt + split
        tl.store(Maxes + stat_offsets, m, mask=row_mask)
        tl.store(Sums + stat_offsets, s_sum, mask=row_mask)
    else:
        tl.store(Out + out_offsets, (acc / s_sum[:, None]).to(Out.dtype.element_ty), mask=row_mask[:, None])


def paged_attention(q: Tensor, k_cache: Tensor, v_cache: Tensor, block_tables: Tensor, context_lens: Tensor) -> Tensor:
    """Attention for T new tokens per sequence whose k/v are already in the cache.

    q [B, heads, T, D]; k/v_cache [num_blocks, kv_heads, block_size, D] for one layer;
    block_tables [B, max_blocks] int32; context_lens [B] int32, tokens cached before the new ones.
    Returns [B, heads, T, D].
    """
    B, H, T, D = q.shape
    _, H_kv, block_size, _ = k_cache.shape
    group = H // H_kv
    q = q.contiguous()

    # Split sequences until there are ~2 programs per GPU core, so small batches don't leave it idle.
    # Larger batches already have enough programs and skip the split (and its merge) entirely.
    # Depends only on the batch size, not on lengths, so a CUDA graph captured for a batch size stays valid.
    splits = triton.cdiv(2 * _num_sms(q.device), B * H_kv)

    if splits == 1:
        out = torch.empty_like(q)
        maxes = sums = out  # unused
        out_strides = (*out.stride()[:3], 0)
        stat_strides = (0, 0, 0)
    else:
        out = torch.empty(B, H, T, splits, D, device=q.device, dtype=torch.float32)
        maxes = torch.empty(B, H, T, splits, device=q.device, dtype=torch.float32)
        sums = torch.empty_like(maxes)
        out_strides = out.stride()[:4]
        stat_strides = maxes.stride()[:3]

    _paged_attention_kernel[(B, H_kv, splits)](
        q, k_cache, v_cache, out, maxes, sums, block_tables, context_lens, D**-0.5, splits,
        *q.stride()[:3],
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        *out_strides,
        *stat_strides,
        GROUP=group,
        QUERY_LEN=T,
        ROWS_PAD=max(16, triton.next_power_of_2(T * group)),
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        # fp32 inputs would otherwise be rounded to tf32 inside tl.dot
        PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
        SPLIT=splits > 1,
    )
    if splits > 1:
        partial = out
        out = torch.empty_like(q)
        _merge_splits_kernel[(B, H * T)](
            partial, maxes, sums, out, splits, T,
            *partial.stride()[:4],
            *maxes.stride()[:3],
            *out.stride()[:3],
            SPLITS_PAD=triton.next_power_of_2(splits),
            HEAD_DIM=D,
        )
    return out


@triton.jit
def _merge_splits_kernel(
    Partial, Maxes, Sums, Out, splits, query_len,
    stride_pb, stride_ph, stride_pt, stride_ps,
    stride_mb, stride_mh, stride_mt,
    stride_ob, stride_oh, stride_ot,
    SPLITS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Combine per-split softmax results for one (sequence, head, new token), rescaling each to the
    overall max, the same correction the main kernel applies block by block. One launch instead of ~9 torch ops."""
    b = tl.program_id(0)
    h = tl.program_id(1) // query_len
    t = tl.program_id(1) % query_len
    s = tl.arange(0, SPLITS_PAD)
    d = tl.arange(0, HEAD_DIM)
    in_range = s < splits
    stats = b * stride_mb + h * stride_mh + t * stride_mt + s
    maxes = tl.load(Maxes + stats, mask=in_range, other=float("-inf"))
    sums = tl.load(Sums + stats, mask=in_range, other=0.0)
    partial_offsets = b * stride_pb + h * stride_ph + t * stride_pt + s[:, None] * stride_ps + d[None, :]
    partial = tl.load(Partial + partial_offsets, mask=in_range[:, None], other=0.0)

    weight = tl.exp(maxes - tl.max(maxes, 0))  # finite max: split 0 always holds key 0, visible to every token
    out = tl.sum(partial * weight[:, None], 0) / tl.sum(sums * weight, 0)
    tl.store(Out + b * stride_ob + h * stride_oh + t * stride_ot + d, out.to(Out.dtype.element_ty))


@functools.cache
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
