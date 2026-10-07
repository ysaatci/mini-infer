import pytest

from mini_infer.draft_policy import AdaptiveDraftPolicy, expected_tokens


@pytest.mark.parametrize("k, a", [(0, 0.5), (2, 0.0), (2, 0.6), (4, 0.9), (3, 1.0)])
def test_expected_tokens_is_the_geometric_series(k, a):
    assert expected_tokens(k, a) == pytest.approx(sum(a**i for i in range(k + 1)))


def policy_with_times(seconds_by_k: dict[int, float], acceptance: float, batch_size: int) -> AdaptiveDraftPolicy:
    policy = AdaptiveDraftPolicy(max_draft_tokens=4, prior_acceptance=acceptance)
    for k, seconds in seconds_by_k.items():
        policy.record_step(k, batch_size, seconds)
    return policy


def test_speculates_when_drafting_is_cheap_relative_to_its_gain():
    # One request at 70% acceptance: k = 2 gives 2.19 tokens per 22 ms, plain decode 1 per 16 ms.
    policy = policy_with_times({0: 0.016, 1: 0.019, 2: 0.022, 3: 0.026, 4: 0.030}, acceptance=0.7, batch_size=1)
    assert policy.choose(["a"]) > 0


def test_falls_back_to_plain_decode_when_steps_get_expensive():
    # A busy GPU: every drafted token adds as much time as a whole plain step.
    policy = policy_with_times({0: 0.030, 1: 0.060, 2: 0.090, 3: 0.120, 4: 0.150}, acceptance=0.7, batch_size=4)
    assert policy.choose(["a", "b", "c", "d"]) == 0


def test_never_picks_an_unmeasured_k():
    policy = policy_with_times({0: 0.020}, acceptance=0.99, batch_size=1)
    assert policy.choose(["a"]) == 0


def test_acceptance_is_tracked_per_request():
    policy = AdaptiveDraftPolicy(prior_acceptance=0.5, smoothing=0.5)
    policy.record_acceptance("easy", accepted=4, drafted=4)
    policy.record_acceptance("hard", accepted=0, drafted=4)
    assert policy.acceptance["easy"] > policy.acceptance["hard"]
