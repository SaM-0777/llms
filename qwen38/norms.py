import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        bias: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.embedding_size = dim
        self.weights = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = self._norm(x.float())
        x = x * self.weights.float()
        if self.bias is not None:
            x = x + self.bias
        return x.to(dtype)


class RMSNormGated(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        x_f = self._norm(x_f)
        x_f = x_f * self.weights.float()
        x_f = x_f * F.silu(gate.float())
        return x_f.to(dtype)


def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x * torch.rsqrt(x.pow(2).sum(dim=dim, keepdim=True) + eps)
