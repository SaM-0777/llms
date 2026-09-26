import math
import torch
import torch.nn as nn

from .config import ModelConfig
from .rope import apply_rotary_emb


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.d_out = cfg.num_heads * cfg.head_dim

        self.W_q = nn.Linear(
            cfg.embedding_dim,
            self.d_out,
            bias=False,
            dtype=cfg.dtype,
        )
        self.W_k = nn.Linear(
            cfg.embedding_dim,
            self.num_kv_heads * cfg.head_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.W_v = nn.Linear(
            cfg.embedding_dim,
            self.num_kv_heads * cfg.head_dim,
            bias=False,
            dtype=cfg.dtype,
        )

        self.out_proj = nn.Linear(
            self.d_out,
            cfg.embedding_dim,
            bias=False,
            dtype=cfg.dtype,
        )

        # kv cache
        self.register_buffer(
            "cache_k",
            torch.zeros(
                cfg.max_batch_size,
                cfg.max_sequence_length,
                self.num_kv_heads,
                self.head_dim,
                dtype=cfg.dtype,
            ),
            persistent=False,
        )

        self.register_buffer(
            "cache_v",
            torch.zeros(
                cfg.max_batch_size,
                cfg.max_sequence_length,
                self.num_kv_heads,
                self.head_dim,
                dtype=cfg.dtype,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        start_pos: int,
        mask: torch.Tensor | None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        B, seqlen, _ = x.shape
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)

        q = q.view(B, seqlen, self.num_heads, self.head_dim)
        k = k.view(B, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(B, seqlen, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freq_cis)

        if use_kv_cache:
            self.cache_k[:B, start_pos : start_pos + seqlen] = k  # type: ignore
            self.cache_v[:B, start_pos : start_pos + seqlen] = v  # type: ignore

            keys = self.cache_k[:B, : start_pos + seqlen]  # type: ignore
            values = self.cache_v[:B, : start_pos + seqlen]  # type: ignore
        else:
            keys = k
            values = v

        keys = keys.repeat_interleave(self.n_rep, dim=2)
        values = values.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(B, seqlen, -1)
        return self.out_proj(output)
