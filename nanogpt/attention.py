import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, embd_dim = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, seq_len, self.n_head, embd_dim // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, seq_len, self.n_head, embd_dim // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, seq_len, self.n_head, embd_dim // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, seq_len, embd_dim)
        y = self.resid_dropout(self.c_proj(y))
        return y
