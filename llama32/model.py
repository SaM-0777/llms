import functools
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer

from .config import ModelConfig
from .rope import compute_rope_params
from .norms import RMSNorm
from .attention import Attention
from .feed_forward import FeedForward


class Block(GradientCheckpointingLayer):
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
        freq_cis: tuple[torch.Tensor, torch.Tensor],
        start_pos: int,
        mask: torch.Tensor | None,
    ):
        h = x + self.attention(self.attention_norm(x), freq_cis, start_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLama3_xs(nn.Module):
    main_input_name = "x"
    supports_gradient_checkpointing = True

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.embedding_dim,
            dtype=cfg.dtype,
        )

        self.norm = RMSNorm(embedding_dim=cfg.embedding_dim, eps=cfg.norm_eps)
        self.blocks = nn.ModuleList([Block(i, cfg) for i in range(cfg.num_blocks)])

        self.out_head = nn.Linear(
            cfg.embedding_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

        cos, sin = compute_rope_params(
            dim=cfg.head_dim,
            context_length=cfg.max_sequence_length,
            theta_base=cfg.rope_theta,
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs=None,
        every_n_layers: int = 1,
        offload: bool = False,
    ):
        if not self.supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing."
            )

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        if offload:
            raise NotImplementedError(f"gradient checkpoint offload not implemente")
        else:
            checkpoint_func = checkpoint

        gradient_checkpointing_func = functools.partial(
            checkpoint_func, **gradient_checkpointing_kwargs
        )
        self._set_gradient_checkpointing(
            enable=True,
            gradient_checkpointing_func=gradient_checkpointing_func,
            every_n_layers=every_n_layers,
        )

    def gradient_checkpoinint_disable(self):
        self._set_gradient_checkpointing(enable=False)

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable[..., Any] = checkpoint,
        every_n_layers: int = 1,
    ):
        is_gradient_checkpointing_set = False
        layer_index = 0

        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                setattr(
                    module, "_gradient_checkpointing_func", gradient_checkpointing_func
                )

                if enable and isinstance(module, GradientCheckpointingLayer):
                    setattr(
                        module,
                        "gradient_checkpointing",
                        layer_index % every_n_layers == 0,
                    )
                    layer_index += 1
                else:
                    setattr(module, "gradient_checkpointing", enable)
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, seqlen = x.shape
        h = self.embedding(x)
        # cos, sin = self.freq_cis
        cos, sin = self.cos.to(h.device), self.sin.to(h.device)
        freq_cis = (
            cos[start_pos : start_pos + seqlen],  # type: ignore
            sin[start_pos : start_pos + seqlen],  # type: ignore
        )

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
        logits = self.out_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ):
        assert (
            len(x.shape) == 2
        ), "Input should be a 2D array of token ids of shape [B, total_token_ids]"

        total_tokens = x.size(1)
        context_length = self.cfg.max_sequence_length
        assert total_tokens <= context_length, (
            f"Input has {total_tokens} tokens, "
            f"but model context length is {context_length}"
        )

        for _ in range(max_new_tokens):
            x_context = x if x.size(1) <= context_length else x[:, -context_length:]

            logits, _ = self(x=x_context)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, token_id_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                token_id_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, token_id_next), dim=1)
            if eos_id is not None and token_id_next.item() == eos_id:
                break

        return x
