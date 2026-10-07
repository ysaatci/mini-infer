import functools

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _paged_decode_kernel(
    Q, K, V, Out, Maxes, Sums, BlockTables, SeqLens, scale, blocks_per_split,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kt,
    stride_tb,
    stride_ob, stride_oh, stride_os,
    stride_mb, stride_mh,
    GROUP: tl.constexpr,  # query heads per k/v head
    GROUP_PAD: tl.constexpr,  # tl.dot needs at least 16 rows
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PRECISION: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # One program per (sequence, k/v head, split). It serves all query heads sharing that k/v head,
    # so each block of keys/values is loaded once and used GROUP times. Each split covers a range of
    # the sequence's blocks, so a short batch still gives the GPU enough programs to run in parallel.
    b = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
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

    start = split * blocks_per_split
    end = tl.minimum(start + blocks_per_split, tl.cdiv(seq_len, BLOCK_SIZE))
    for j in range(start, end):
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

    out_offsets = b * stride_ob + heads[:, None] * stride_oh + split * stride_os + d[None, :]
    if SPLIT:
        # Partial result for this range, merged across splits afterwards. An empty range (short
        # sequence, late split) stores max -inf and sum 0, so it gets zero weight in the merge.
        tl.store(Out + out_offsets, acc, mask=head_mask[:, None])
        stat_offsets = b * stride_mb + heads * stride_mh + split
        tl.store(Maxes + stat_offsets, m, mask=head_mask)
        tl.store(Sums + stat_offsets, s_sum, mask=head_mask)
    else:
        tl.store(Out + out_offsets, (acc / s_sum[:, None]).to(Out.dtype.element_ty), mask=head_mask[:, None])


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

    # Split sequences until there are ~2 programs per GPU core, so small batches don't leave it idle.
    # Larger batches already have enough programs and skip the split (and its merge) entirely.
    max_blocks = block_tables.shape[1]
    splits = min(max_blocks, triton.cdiv(2 * _num_sms(q.device), B * H_kv))
    blocks_per_split = triton.cdiv(max_blocks, splits)
    splits = triton.cdiv(max_blocks, blocks_per_split)

    if splits == 1:
        out = torch.empty_like(q)
        maxes = sums = out  # unused
        out_strides = (out.stride(0), out.stride(1), 0)
        stat_strides = (0, 0)
    else:
        out = torch.empty(B, H, splits, D, device=q.device, dtype=torch.float32)
        maxes = torch.empty(B, H, splits, device=q.device, dtype=torch.float32)
        sums = torch.empty_like(maxes)
        out_strides = (out.stride(0), out.stride(1), out.stride(2))
        stat_strides = (maxes.stride(0), maxes.stride(1))

    _paged_decode_kernel[(B, H_kv, splits)](
        q, k_cache, v_cache, out, maxes, sums, block_tables, seq_lens, D**-0.5, blocks_per_split,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        *out_strides,
        *stat_strides,
        GROUP=group,
        GROUP_PAD=max(16, triton.next_power_of_2(group)),
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        # fp32 inputs would otherwise be rounded to tf32 inside tl.dot
        PRECISION="ieee" if q.dtype == torch.float32 else "tf32",
        SPLIT=splits > 1,
    )
    if splits > 1:
        partial = out
        out = torch.empty_like(q)
        _merge_splits_kernel[(B, H)](
            partial, maxes, sums, out, splits,
            partial.stride(0), partial.stride(1), partial.stride(2),
            maxes.stride(0), maxes.stride(1),
            out.stride(0), out.stride(1),
            SPLITS_PAD=triton.next_power_of_2(splits),
            HEAD_DIM=D,
        )
    return out.view(B, H, 1, D)


@triton.jit
def _merge_splits_kernel(
    Partial, Maxes, Sums, Out, splits,
    stride_pb, stride_ph, stride_ps,
    stride_mb, stride_mh,
    stride_ob, stride_oh,
    SPLITS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Combine per-split softmax results for one (sequence, head), rescaling each to the overall max,
    the same correction the main kernel applies block by block. One launch instead of ~9 torch ops."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, SPLITS_PAD)
    d = tl.arange(0, HEAD_DIM)
    in_range = s < splits
    maxes = tl.load(Maxes + b * stride_mb + h * stride_mh + s, mask=in_range, other=float("-inf"))
    sums = tl.load(Sums + b * stride_mb + h * stride_mh + s, mask=in_range, other=0.0)
    partial = tl.load(Partial + b * stride_pb + h * stride_ph + s[:, None] * stride_ps + d[None, :], mask=in_range[:, None], other=0.0)

    weight = tl.exp(maxes - tl.max(maxes, 0))  # finite max: split 0 always has at least one token
    out = tl.sum(partial * weight[:, None], 0) / tl.sum(sums * weight, 0)
    tl.store(Out + b * stride_ob + h * stride_oh + d, out.to(Out.dtype.element_ty))


@functools.cache
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
