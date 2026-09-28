from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    vocab_size: int = -1
    embedding_dim: int = 640
    max_sequence_length: int = 32768
    head_dim: int = 256
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-6
    num_heads: int = 4
    num_kv_heads: int = 1
    num_blocks: int = 18
    max_batch_size: int = 32
    hidden_dim: int = 2048
    use_kv_cache: bool = False

    seed: int = 42

    tokenizer_name: str = "gpt2"
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.bfloat16
    torch_compile: bool = False

    # training configs
    output_dir: str | Path = "runs/llama32"
    data_dir: str | Path = "data/fineweb_1.25B"
    resume_from: str | Path | None = None
    num_dataset_workers: int | None = 0
    learning_rate: float = 1e-4
    min_lr: float = 5e-5
    adam_epsilon: float = 1e-9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    max_iters: int = 150_000
    warmup_steps: int = 1000
    batch_size: int = 32
    block_size: int = 128
    gradient_accumulation_steps: int = 32
    eval_intervals: int = 1000
    eval_iters: int = 200
    log_interval: int = 10
    save_interval: int = 10000

    # Project configs
    project_name: str = "LLama_32"
    wandb_log: bool = False
