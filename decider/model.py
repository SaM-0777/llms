from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import cast
import string

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5TextConfig,
    AutoTokenizer,
    Qwen3VLProcessor,
)
from safetensors.torch import load_file
from transformers.masking_utils import create_recurrent_attention_mask

from decider.encoder import MAX_OPTIONS, Encoded, Encoder, left_pad

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2b48083dfa97c3cbf0220cf1a5e6cffe0a511157"


@dataclass(frozen=True)
class PreparedBatch:
    inputs: dict[str, torch.Tensor]
    counts: tuple[int, ...]
    input_tokens: int


class DecisionModel(nn.Module):
    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str = REVISION,
        attention_mode: str = "noncausal_full_attention",
        checkpoint: str | Path | None = None,
        dtype: torch.dtype | None = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
        config=None,
        train: bool = False,
        gradient_checkpointing: bool = False,
        temperature: float | None = 0.0,
        pooling: str = "last",
        max_length: int = 2048,
    ) -> None:
        super().__init__()
        self.device = device
        self.model_id = model_id
        self.revision = revision
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        self.decision_candidates = list(string.ascii_uppercase) + [
            "".join(pair)
            for pair in itertools.product(string.ascii_uppercase, repeat=2)
        ]
        self.decision_codes = [
            code
            for code in self.decision_candidates
            if len(self.tokenizer.encode(code, add_special_tokens=False)) == 1
        ][:MAX_OPTIONS]
        self.decision_token_ids = [
            self.tokenizer.encode(code, add_special_tokens=False)[0]
            for code in self.decision_codes
        ]
        if len(set(self.decision_token_ids)) != MAX_OPTIONS:
            raise ValueError(
                "Tokenizer must provide 255 distinct single-token answer codes."
            )

        self.pooling = pooling
        self.attention_mode = attention_mode
        self.processor = cast(
            Qwen3VLProcessor,
            AutoProcessor.from_pretrained(
                str(checkpoint) if checkpoint else self.model_id,
                revision=None if checkpoint else self.revision,
            ),
        )

        self.encoder = Encoder(
            processor=self.processor, decision_codes=self.decision_codes
        )

        if not dtype:
            dtype = torch.float32

        if checkpoint is None:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.model_id,
                revision=revision,
                dtype=dtype,
                attn_implementation="sdpa",
                cache_dir=None,
            )
            config = cast(Qwen3_5TextConfig, model.config.text_config)
            self.readout = torch.nn.Linear(
                config.hidden_size, MAX_OPTIONS, bias=False, dtype=dtype
            )
            with torch.no_grad():
                self.readout.weight.copy_(model.lm_head.weight[self.decision_token_ids])
            self.backbone = model.model
            del model
        else:
            self.backbone = Qwen3_5Model.from_pretrained(
                str(checkpoint), dtype=dtype, attn_implementation="sdpa"
            )
            config = cast(Qwen3_5TextConfig, self.backbone.config.text_config)
            self.readout = torch.nn.Linear(
                config.hidden_size, MAX_OPTIONS, bias=False, dtype=dtype
            )
            self.readout.load_state_dict(
                load_file(str(Path(checkpoint) / "readout.safetensors"))
            )

        if self.attention_mode == "noncausal_full_attention":
            self.noncausal_full_attention()

        self.backbone.requires_grad_(False)
        self.readout.requires_grad_(True)

        if train and gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.temperature = temperature or 1.0
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be positive and finite.")

        self.to(self.device)
        self.train(False)

    def noncausal_full_attention(self):
        text_model = self.backbone.language_model

        if text_model.config._attn_implementation != "sdpa":
            raise ValueError("The attention ablation requires SDPA")

        def mask_inputs(module, args, kwargs):
            if args:
                raise ValueError("The attention ablation requires keyword inputs")
            if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
                raise ValueError("Noncausal classification does not support a KV cache")
            embeddings = kwargs.get("inputs_embeds")
            if embeddings is None:
                embeddings = module.embed_tokens(kwargs["input_ids"])
            padding = kwargs.get("attention_mask")
            if padding is None:
                padding = torch.ones(
                    embeddings.shape[:2], device=embeddings.device, dtype=torch.bool
                )
            if not isinstance(padding, torch.Tensor) or padding.ndim != 2:
                raise ValueError("Expected the processor's 2D padding mask")
            if (
                padding.shape != embeddings.shape[:2]
                or not padding.bool().any(dim=-1).all()
            ):
                raise ValueError("Padding mask must match the complete nonempty input")
            kwargs["attention_mask"] = {
                "full_attention": padding[:, None, None, :].bool(),
                "linear_attention": create_recurrent_attention_mask(
                    config=module.config,
                    inputs_embeds=embeddings,
                    attention_mask=padding,
                ),
            }
            return args, kwargs

        text_model.register_forward_pre_hook(mask_inputs, with_kwargs=True)

    def forward(self, chunks: Sequence[Encoded]) -> list[list[torch.Tensor]]:
        inputs = {
            key: value.to(self.device)
            for key, value in left_pad(chunks, self.encoder.pad_id).items()
        }
        hidden = self.backbone(**inputs, use_cache=False).last_hidden_state
        offsets = [inputs["input_ids"].shape[1] - chunk.length for chunk in chunks]
        rows = torch.cat(
            [torch.full((len(c.keys),), i) for i, c in enumerate(chunks)]
        ).to(self.device)
        cols = torch.cat([c.markers + offsets[i] for i, c in enumerate(chunks)]).to(
            self.device
        )
        logits = self.readout(hidden[rows, cols]).float()
        out, cursor = [], 0
        for chunk in chunks:
            out.append(
                [logits[cursor + j, :count] for j, count in enumerate(chunk.counts)]
            )
            cursor += len(chunk.keys)
        return out
