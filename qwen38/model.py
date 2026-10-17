import torch
from torch import nn
import torch.nn.functional as F

from qwen38.config import Config
from qwen38.rope import calculate_rope_params
from qwen38.norms import RMSNorm
from qwen38.mlp import FeedForward
from qwen38.attention import GatedDeltaNetAttention, FullAttention


class Layer(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int) -> None:
        super().__init__()

        self.cfg = cfg
        self.layer_idx = layer_idx
        self.layer_type = cfg.layers[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attention = GatedDeltaNetAttention(cfg=cfg, layer_idx=layer_idx)
        elif self.layer_type == "full_attention":
            self.full_attention = FullAttention(cfg=cfg, layer_idx=layer_idx)
        else:
            raise ValueError(
                f"Invalid layer type at layer index {layer_idx}."
                "Supported values 'linear_attention', 'full_attention'"
            )

        self.feed_forward = FeedForward(cfg=cfg)
        self.attention_norm = RMSNorm(
            dim=cfg.embedding_dim,
            eps=cfg.norm_eps,
        )
        self.ffn_norm = RMSNorm(
            dim=cfg.embedding_dim,
            eps=cfg.norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        start_pos: int,
        causal_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        x_res = x
        x = self.attention_norm(x)

        if self.layer_type == "linear_attention":
            x = self.linear_attention(x, padding_mask)  # Padding mask, NOT Causal mask
        elif self.layer_type == "full_attention":
            x = self.full_attention(
                x, freqs_cis, start_pos, causal_mask
            )  # Causal mask, NOT Padding mask
        else:
            raise RuntimeError(
                f"Invalid layer type {self.layer_type} at layer {self.layer_idx}"
            )

        x = x_res + x

        out = self.ffn_norm(x)
        out = self.feed_forward(out)
        out = x + out

        return out


class Qwen38(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        if not cfg.vocab_size:
            raise ValueError(f"Provide valid vocab size")

        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.embedding_dim = cfg.embedding_dim

        self.embedding = nn.Embedding(
            cfg.vocab_size, cfg.embedding_dim, dtype=cfg.dtype
        )

        cos, sin = calculate_rope_params(
            dim=cfg.head_dim,
            context_len=cfg.max_seqlen,
            theta_base=cfg.rope_theta,
            device=cfg.device,
            dtype=torch.float32,
        )

        self.layers = nn.ModuleList(
            [Layer(cfg=cfg, layer_idx=idx) for idx in range(len(cfg.layers))]
        )

        self.norm = RMSNorm(dim=cfg.embedding_dim)
        self.out_head = nn.Linear(
            cfg.embedding_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        start_pos: int = 0,
    ):
        _, seqlen = x.shape
        h = self.embedding(x)

        cos, sin = self.cos.to(h.device), self.sin.to(h.device)
        freqs_cis = (
            cos[start_pos : start_pos + seqlen],  # type: ignore
            sin[start_pos : start_pos + seqlen],  # type: ignore
        )

        causal_mask = None
        if seqlen > 1:
            causal_mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=x.device), causal_mask]
            ).to(h.dtype)

        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.bool)

        for i, decoder_layer in enumerate(self.layers):
            h = decoder_layer(h, freqs_cis, start_pos, causal_mask, padding_mask)

        h = self.norm(h)
        logits = self.out_head(h)  # vocab_size

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss
