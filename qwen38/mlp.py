import torch
import torch.nn as nn

from qwen38.config import Config


class FeedForward(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.cfg = cfg

        self.gate_proj = nn.Linear(
            in_features=cfg.embedding_dim,
            out_features=cfg.ffn_dim,
            dtype=cfg.dtype,
            bias=False,
        )
        self.up_proj = nn.Linear(
            in_features=cfg.embedding_dim,
            out_features=cfg.ffn_dim,
            dtype=cfg.dtype,
            bias=False,
        )
        self.down_proj = nn.Linear(
            in_features=cfg.ffn_dim,
            out_features=cfg.embedding_dim,
            dtype=cfg.dtype,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_proj(nn.SiLU(self.gate_proj(x)) * self.up_proj(x))
        return out
