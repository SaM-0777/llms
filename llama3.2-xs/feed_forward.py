import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.f1 = nn.Linear(
            cfg.embedding_dim,
            cfg.hidden_dim,
            dtype=cfg.dtype,
            bias=False,
        )
        self.f2 = nn.Linear(
            cfg.hidden_dim, cfg.embedding_dim, dtype=cfg.dtype, bias=False
        )
        self.f3 = nn.Linear(
            cfg.embedding_dim,
            cfg.hidden_dim,
            dtype=cfg.dtype,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.f1(x)
        x1 = F.silu(x1)
        x3 = self.f3(x)
        x = x1 * x3
        x = self.f2(x)
        return x
