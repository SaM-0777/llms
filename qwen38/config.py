from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    name: str = "Qwen38"
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda")

    batch_size: int = 64

    tokenizer_name: str = "tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-py-lexical"
    vocab_size: int = -1
    embedding_dim: int = 2048
    num_attention_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 256
    rope_theta: float = 1000000.0
    max_seqlen: int = 32768
    use_cache: bool = False
    norm_eps: float = 1e-4
    ffn_dim: int = 7168
    linear_conv_kernel_dim: int = 4
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    layers: list[str] = field(
        default_factory=lambda: [
            # block_0
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            # block_1
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            # block_2
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            # block_3
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            # block_4
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            # block_5
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            # block_6
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            # block_7
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
    )

    project_name: str = "llm_Qwen38"
