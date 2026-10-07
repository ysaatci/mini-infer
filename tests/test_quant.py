import pytest
import torch
from torch import nn

from mini_infer.quant import Int8Linear, int8_matmul, quantize_weight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_round_trip_error_is_at_most_half_a_step():
    torch.manual_seed(0)
    w = torch.randn(256, 512, device="cuda")
    q, scale = quantize_weight(w)
    error = (q.float() * scale[:, None] - w).abs()
    assert (error <= scale[:, None] / 2 + 1e-6).all()


# Qwen2.5-1.5B shapes: attention q/o (1536 x 1536), MLP up (8960 x 1536) and down (1536 x 8960).
@pytest.mark.parametrize("rows", [1, 5, 64, 300])  # 300: prefill tiles
@pytest.mark.parametrize("n, k", [(1536, 1536), (8960, 1536), (1536, 8960)])
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-4), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_int8_matmul_matches_dequantized_reference(rows, n, k, dtype, tol, with_bias):
    torch.manual_seed(0)
    q, scale = quantize_weight(torch.randn(n, k, device="cuda"))
    x = torch.randn(rows, k, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype) if with_bias else None
    expected = x.float() @ (q.float() * scale[:, None]).T + (bias.float() if with_bias else 0)
    actual = int8_matmul(x, q, scale.to(dtype), bias)
    torch.testing.assert_close(actual.float(), expected, atol=tol * k**0.5, rtol=tol)


def test_int8_linear_on_a_batch_of_prompts_matches_reference():
    # [batch, tokens, features] input, enough rows for the prefill tiles.
    torch.manual_seed(0)
    linear = nn.Linear(1536, 1536, bias=True, device="cuda")
    layer = Int8Linear(linear)
    x = torch.randn(2, 150, 1536, device="cuda")
    expected = x @ (layer.weight.float() * layer.scale[:, None]).T + layer.bias
    torch.testing.assert_close(layer(x), expected, atol=1e-4, rtol=1e-4)
