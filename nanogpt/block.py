import torch
import torch.nn as nn

from attention import CausalAttention


class LayerNorm(nn.Module):
    def __init__(
        self,
        ndim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.layer_norm(x, self.weights.shape, self.weights, self.bias)


class MLP(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(ndim=cfg.n_embd, bias=cfg.bias)
        self.self_cattn = CausalAttention(cfg=cfg)
        self.ln_2 = LayerNorm(ndim=cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg=cfg)

    def forward(self, x):
        x = x + self.self_cattn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
