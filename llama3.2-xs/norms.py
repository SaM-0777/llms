import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self, embdding_dim: int, bias: bool = False, eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(embdding_dim))
        self.bias = nn.Parameter(torch.zeros(embdding_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x + torch.rsqrt(var + self.eps)
        out = x_norm * self.weights.float()
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(dtype)
