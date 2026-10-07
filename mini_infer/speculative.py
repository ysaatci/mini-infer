import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import Tensor

from mini_infer.cuda_graphs import DecodeGraphRunner
from mini_infer.draft_policy import AdaptiveDraftPolicy, DraftPolicy
from mini_infer.model import Qwen2ForCausalLM
from mini_infer.paged_cache import PagedKVPool
from mini_infer.request import Request
from mini_infer.sampling import probabilities


@dataclass(frozen=True)
class SpeculativeConfig:
    draft_model: Qwen2ForCausalLM  # must share the target's vocabulary
    # Picks k each step. Adaptive by default; FixedDraftPolicy(k) reproduces a fixed setting.
    policy: DraftPolicy = field(default_factory=AdaptiveDraftPolicy)
    max_draft_tokens: int = 4  # verify graphs are recorded for k = 1 .. this
    max_batch_size: int = 16  # never speculate above this many requests, so no graphs are recorded for more


def accept(draft_tokens: Tensor, draft_probs: Tensor, target_probs: Tensor, generator: torch.Generator | None = None):
    """Speculative sampling (Leviathan et al. 2023): the output follows the target's distribution exactly.

    draft_tokens [B, k] were sampled from draft_probs [B, k, V]; target_probs [B, k + 1, V] are the
    target's distributions at the same positions plus one more. Drafted token i is accepted with
    probability min(1, p(d_i) / q(d_i)); the first rejection ends the run. The next token then comes from
    the leftover distribution max(0, p - q), renormalized, which corrects for the draft's bias, or, if
    every draft was accepted, from the target's distribution at position k.
    Greedy rows are one-hot on both sides, so this reduces to "accept while equal, then the target's token".

    Returns (accepted drafts per row [B], next token per row [B]).
    """
    B, k = draft_tokens.shape
    p = target_probs[:, :k].gather(-1, draft_tokens[..., None]).squeeze(-1)
    q = draft_probs.gather(-1, draft_tokens[..., None]).squeeze(-1)  # > 0: the token was sampled from it
    u = torch.rand(B, k, device=p.device, generator=generator)
    accepted = (u * q < p).int().cumprod(-1)  # u < p / q, without dividing; 1s up to the first rejection
    num_accepted = accepted.sum(-1)

    rows = torch.arange(B, device=p.device)
    target_next = target_probs[rows, num_accepted]
    draft_next = draft_probs[rows, num_accepted.clamp(max=k - 1)] * (num_accepted < k)[:, None]
    leftover = (target_next - draft_next).clamp(min=0)
    total = leftover.sum(-1, keepdim=True)
    # total is 0 only when p equals q exactly, where rejection can't happen; fall back to p just in case.
    leftover = torch.where(total > 0, leftover / total.clamp(min=1e-30), target_next)
    return num_accepted, torch.multinomial(leftover, 1, generator=generator).squeeze(-1)


class SpeculativeDecoder:
    """Draft k tokens per request with a small model, check them all with the target in one forward.

    Decoding is limited by reading the weights, so the target checking k + 1 positions costs about the
    same as generating one token. When the draft guesses well, each target pass yields several tokens.

    The draft keeps its own paged cache, with the same block size and sequence ids as the target's. It
    catches up lazily: before drafting, any tokens it hasn't seen (a new prompt, a request resumed after
    preemption, steps that ran without speculation) are run through it. Rejected tokens are rolled back
    by resetting lengths; their stale k/v are simply overwritten later.

    k can change from step to step (the policy decides), so there is a recorded verify graph per k.
    """

    def __init__(self, target: Qwen2ForCausalLM, target_pool: PagedKVPool, config: SpeculativeConfig, use_cuda_graphs: bool):
        draft = config.draft_model
        if draft.config.vocab_size != target.config.vocab_size:
            raise ValueError("draft and target models must share a vocabulary")
        self.target, self.draft = target, draft
        self.target_pool = target_pool
        self.policy = config.policy
        self.max_draft_tokens = config.max_draft_tokens
        self.max_batch_size = config.max_batch_size
        # As many blocks as the target: the draft never holds more tokens per request than the target.
        self.draft_pool = PagedKVPool(draft.config, target_pool.num_blocks, target_pool.block_size, draft.device, draft.dtype)
        self.draft_graphs = None
        self.verify_graphs: dict[int, DecodeGraphRunner] = {}
        if use_cuda_graphs:
            # One memory pool for all of them: only one graph runs at a time.
            memory_pool = torch.cuda.graph_pool_handle()
            self.draft_graphs = DecodeGraphRunner(draft, self.draft_pool, config.max_batch_size, memory_pool=memory_pool)
            self.verify_graphs = {
                k: DecodeGraphRunner(target, target_pool, config.max_batch_size, query_len=k + 1, memory_pool=memory_pool)
                for k in range(1, config.max_draft_tokens + 1)
            }
        self.num_drafted = 0
        self.num_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / max(self.num_drafted, 1)

    def release(self, seq_id: str) -> None:
        self.draft_pool.free(seq_id)
        self.policy.release(seq_id)

    @torch.inference_mode()
    def step(self, requests: list[Request], k: int) -> list[list[int]] | None:
        """New tokens per request (1 to k + 1 each), with both caches advanced to match.
        None if the draft cache has no room this step; the caller then decodes normally.
        The target pool must already have room for k + 1 more tokens per request."""
        seq_ids = [r.id for r in requests]
        if not self._catch_up_draft(requests, k):
            return None
        # Timed after catch-up: the policy compares the cost of a step with k drafts against plain decode.
        start = time.perf_counter()
        params = [r.params for r in requests]
        all_greedy = all(p.temperature == 0 for p in params)
        context = [self.target_pool.lengths[s] for s in seq_ids]  # both caches hold all but the last token

        # Draft k tokens, one forward each, plus one more forward that only stores the k-th token's k/v.
        current = torch.tensor([[r.output_ids[-1]] for r in requests], device=self.draft_pool.k.device)
        draft_tokens, draft_probs = [], []
        for i in range(k + 1):
            logits = self._forward(self.draft, self.draft_pool, self.draft_graphs, current, seq_ids)[:, -1]
            if i == k:
                break
            q = probabilities(logits, params)
            # One-hot rows (greedy) always give their top token; argmax gets it without sampling 152k entries.
            current = q.argmax(-1, keepdim=True) if all_greedy else torch.multinomial(q, 1)
            draft_tokens.append(current)
            draft_probs.append(q)
        draft_tokens = torch.cat(draft_tokens, dim=1)  # [B, k]

        # Verify: the last accepted token plus the k drafts, all in one target forward.
        verify_input = torch.cat([torch.tensor([[r.output_ids[-1]] for r in requests], device=draft_tokens.device), draft_tokens], dim=1)
        verify_graphs = self.verify_graphs.get(k)
        logits = self._forward(self.target, self.target_pool, verify_graphs, verify_input, seq_ids)  # [B, k+1, V]
        B, T, V = logits.shape
        target_probs = probabilities(logits.reshape(B * T, V), [p for p in params for _ in range(T)]).view(B, T, V)
        num_accepted, next_token = accept(draft_tokens, torch.stack(draft_probs, dim=1), target_probs)

        # Roll both caches back to the accepted length. Writes past it are stale and get overwritten later.
        num_accepted, next_token, drafts = num_accepted.tolist(), next_token.tolist(), draft_tokens.tolist()  # one sync
        new_tokens = []
        for i, seq_id in enumerate(seq_ids):
            n = num_accepted[i]
            self.target_pool.lengths[seq_id] = self.draft_pool.lengths[seq_id] = context[i] + n + 1
            new_tokens.append(drafts[i][:n] + [next_token[i]])
            self.policy.record_acceptance(seq_id, n, k)
        self.policy.record_step(k, B, time.perf_counter() - start)
        self.num_drafted += k * B
        self.num_accepted += sum(num_accepted)
        return new_tokens

    def _catch_up_draft(self, requests: list[Request], k: int) -> bool:
        """Bring each draft cache to "every known token but the last", and reserve room for drafting.

        Requests missing the same number of tokens catch up together in one forward. After the policy
        switches speculation off and back on, every request is behind by the same few tokens, so this
        is one small batched forward instead of one per request.
        """
        groups: dict[tuple[int, bool], list[Request]] = defaultdict(list)
        for request in requests:
            have = self.draft_pool.lengths.get(request.id, 0)
            missing = len(request.all_ids) - 1 - have
            if not self.draft_pool.reserve(request.id, missing + k + 1):
                return False
            if missing:
                # Fresh caches and ones with history take different attention paths, so they don't mix.
                groups[(missing, have > 0)].append(request)
        for (missing, _), group in groups.items():
            seq_ids = [r.id for r in group]
            have = [self.draft_pool.lengths[s] for s in seq_ids]
            tokens = torch.tensor([r.all_ids[h : h + missing] for r, h in zip(group, have)], device=self.draft_pool.k.device)
            cache = self.draft_pool.view(seq_ids)
            positions = cache.lengths_t[:, None].long() + torch.arange(missing, device=tokens.device)
            self.draft(tokens, positions, cache, last_token_only=True)
            cache.advance(missing)
        return True

    @staticmethod
    def _forward(model, pool: PagedKVPool, graphs: DecodeGraphRunner | None, tokens: Tensor, seq_ids: list[str]) -> Tensor:
        """tokens [B, T] for these sequences -> logits [B, T, V], and advance the pool by T."""
        T = tokens.shape[1]
        if graphs is not None:
            logits = graphs.run(tokens, seq_ids)
        else:
            cache = pool.view(seq_ids)
            positions = cache.lengths_t[:, None].long() + torch.arange(T, device=tokens.device)
            logits = model(tokens, positions, cache, last_token_only=T == 1)
        pool.advance(seq_ids, T)
        return logits
