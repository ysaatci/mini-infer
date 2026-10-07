import torch

from mini_infer.speculative import accept

VOCAB, K, TRIALS = 8, 3, 200_000


def total_variation(samples: torch.Tensor, expected: torch.Tensor) -> float:
    empirical = torch.bincount(samples, minlength=VOCAB).float() / len(samples)
    return 0.5 * (empirical - expected).abs().sum().item()


def test_output_follows_the_target_distribution():
    # Made-up target (p) and draft (q) distributions per position, deliberately different.
    torch.manual_seed(0)
    p = torch.distributions.Dirichlet(torch.ones(VOCAB)).sample((K + 1,))
    q = torch.distributions.Dirichlet(torch.ones(VOCAB)).sample((K,))
    draft_tokens = torch.stack([torch.multinomial(q[i], TRIALS, replacement=True) for i in range(K)], dim=1)

    num_accepted, next_token = accept(draft_tokens, q.expand(TRIALS, K, VOCAB), p.expand(TRIALS, K + 1, VOCAB))

    # Emitted tokens are the accepted drafts, then next_token.
    first = torch.where(num_accepted >= 1, draft_tokens[:, 0], next_token)
    assert total_variation(first, p[0]) < 0.01
    # Given the first draft was accepted, the second token must follow the target's next distribution.
    went_on = num_accepted >= 1
    second = torch.where(num_accepted[went_on] >= 2, draft_tokens[went_on, 1], next_token[went_on])
    assert total_variation(second, p[1]) < 0.01


def test_greedy_accepts_while_equal_then_takes_the_target_token():
    one_hot = lambda tokens: torch.nn.functional.one_hot(torch.tensor(tokens), VOCAB).float()[None]
    draft = torch.tensor([[1, 2, 3]])
    target = one_hot([1, 2, 5, 6])  # agrees on two drafts, then picks 5
    num_accepted, next_token = accept(draft, one_hot([1, 2, 3]), target)
    assert num_accepted.item() == 2 and next_token.item() == 5
