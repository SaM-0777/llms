import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import compute_rope_params
from .norms import RMSNorm
from .attention import Attention
from .feed_forward import FeedForward


class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = cfg.num_heads
        self.embedding_dim = cfg.embedding_dim
        self.head_dim = cfg.head_dim
        self.attention = Attention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.attention_norm = RMSNorm(embedding_dim=cfg.embedding_dim, eps=cfg.norm_eps)
        self.ffn_norm = RMSNorm(embedding_dim=cfg.embedding_dim, eps=cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None,
    ):
        h = x + self.attention(self.attention_norm(x), freq_cis, start_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLama3_xs(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.embedding_dim,
            dtype=cfg.dtype,
        )

        self.freq_cis = compute_rope_params(
            dim=cfg.head_dim,
            context_length=cfg.max_sequence_length,
            theta_base=cfg.rope_theta,
        )

        self.norm = RMSNorm(embedding_dim=cfg.embedding_dim, eps=cfg.norm_eps)
        self.blocks = nn.ModuleList([Block(i, cfg) for i in range(cfg.num_blocks)])

        self.out_head = nn.Linear(
            cfg.embedding_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, seqlen = x.shape
        h = self.embedding(x)
        self.freq_cis = self.freq_cis.to(h.device)
        freq_cis = self.freq_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=x.device), mask]
            ).to(h.dtype)

        for block in self.blocks:
            h = block(h, freq_cis, start_pos, mask)

        h = self.norm(h)
        logits = self.out_head(h).float()

        loss = None
        if targets is not None:
            targets = targets.float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss
