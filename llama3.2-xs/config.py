from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int = 640
    max_sequence_length: int = 32768
    head_dim: int = 256
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-6
    num_heads: int = 4
    num_kv_heads: int = 1
    num_blocks: int = 18
    max_batch_size: int = 1
    hidden_dim: int = 2048

    seed: int = 42

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.bfloat16

    # training configs
    data_dir: str | Path = "data/fineweb_1.25B"
    resume_from: str | Path | None = None
    learning_rate: float = 1e-4
    min_lr: float = 5e-5
    max_iters: int = 150_000
    warmup_steps: int = 1000
    batch_size: int = 32
    block_size: int = 128
    gradient_accumulation_step: int = 32
    eval_intervals: int = 500
    eval_iters: int = 200
    log_interval: int = 10

    # Project configs
    project_name: str = "LLama_3.2_xs"
