import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Normalize in fp32: squaring bf16 values loses too much precision.
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


class RotaryEmbedding:
    """Encodes position by rotating pairs of q/k dimensions by a position-dependent angle."""

    def __init__(self, head_dim: int, theta: float):
        self.head_dim = head_dim
        self.theta = theta

    def __call__(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        # positions: [B, T] -> cos, sin: [B, 1, T, head_dim]
        exponents = torch.arange(0, self.head_dim, 2, device=positions.device).float() / self.head_dim
        inv_freq = 1.0 / (self.theta**exponents)
        angles = positions[..., None].float() * inv_freq
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos().to(dtype)[:, None], angles.sin().to(dtype)[:, None]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
