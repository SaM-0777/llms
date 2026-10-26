from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Args:
    seed: int = 42
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    model_id: str = "Qwen/Qwen3.5-0.8B"
    batch_size: int = 16
    dataset_id: str = "LocalLLaMA/typed-decisions"
    dataset_name: str | None = "all"
    train_split: str = "train"
    eval_split: str = "test"

    max_seq_length: int = 2048
    full_finetuning: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: str = "all-linear"
    lora_bias: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"

    output_dir: str | Path = "output"
    resume_from_checkpoint: str | None = None
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    max_grad_norm: float | None = 1.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    lr_scheduler_type: str = "cosine"
    brier_weight: float = 0.0
    logging_steps: int = 2
    eval_steps: int = 20
    save_steps: int = 100
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    gradient_checkpointing: bool = False
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: Literal[
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ] = "default"
    report_to: str = "none"
    project_name: str = "Decider"
    run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: [
            "decider",
            "qwen3.5-0.8B",
            "lora",
            "decision-model",
        ]
    )
