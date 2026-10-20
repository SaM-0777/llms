from pathlib import Path
from dataclasses import dataclass, field
import json
import logging
from typing import Literal
from datetime import datetime

import torch
import tyro
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import huggingface_hub as hf
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import (
    get_dataset_split_names,
    load_dataset,
)
import wandb

LOG = logging.getLogger("SFT")


@dataclass
class Args:
    seed: int = 42
    model_id: str = "Qwen/Qwen3.5-4B-Base"
    tokenizer_id: str | None = None
    dataset_id: str = "nvidia/Nemotron-SFT-SWE-v3.5"
    train_split: str = "train"
    test_split: str = "test"
    test_train_ratio: float = 0.1
    dataset_kwargs: dict = field(
        default_factory=lambda: {
            "streaming": True,
        }
    )
    max_seq_length: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"
    lora_target_modules: list[str] | None = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"

    output_dir: str | Path = "output"
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: float = 250
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_seq_length: int = 4096
    gradient_checkpointing: bool = False
    report_to: str = "wandb"
    project_name: str = "Qwen3.5-4B_SWE"
    run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(
        default_factory=lambda: [
            "qwen3.5-4B",
            "sft",
            "qlora",
            "swe",
        ]
    )


def hf_auth():
    try:
        whoami = hf.whoami()
        LOG.info("type=%s name=%s", whoami.get("type"), whoami.get("name"))
    except Exception as e:
        raise RuntimeWarning(f"Authenticate HF to prevent rate limits.")


def configure_wandb(args: Args) -> str | None:
    if args.report_to != "wandb":
        return None

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.project_name}_{timestamp}"

    config = {
        "seed": args.seed,
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "dataset_id": args.dataset_id,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "test_train_ratio": args.test_train_ratio,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_bias": args.lora_bias,
        "bias": args.bias,
        "lora_target_modules": args.lora_target_modules,
        "load_in_4bit": args.load_in_4bit,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_seq_length": args.max_seq_length,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    run = wandb.init(
        project=args.project_name,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags,
        config=config,
        job_type="sft",
        save_code=True,
        reinit=True,
    )

    LOG.info(
        "Initialized W&B: project=%s run=%s",
        args.project_name,
        run.name,
    )

    return run.id


def load_tokenizer(tokenizer_id: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    return tokenizer


def process_example(example: dict):
    for message in example.get("messages", []):
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments")

            if isinstance(arguments, str):
                function["arguments"] = json.loads(arguments)

    return example


def load_datasets(
    dataset_id: str,
    dataset_kwargs: dict,
    train_split: str = "train",
    test_split: str = "test",
    test_train_ratio: float = 0.1,
    seed: int = 42,
):
    if not 0.0 < test_train_ratio < 1.0:
        raise ValueError("train_test_ratio must be between 0 and 1.")

    splits = get_dataset_split_names(dataset_id)

    if train_split not in splits:
        raise ValueError(
            f"Training split '{train_split}' does not exist.",
            f"Available splits: {splits}",
        )

    if test_split in splits:
        train_dataset = load_dataset(dataset_id, split=train_split, **dataset_kwargs)
        test_dataset = load_dataset(dataset_id, split=test_split, **dataset_kwargs)
    else:
        # If split does not exists, then split the train split according to test_train_ratio
        dataset = load_dataset(dataset_id, split=train_split)
        dataset = dataset.shuffle(seed=seed)

        data_split = dataset.train_test_split(test_size=test_train_ratio, seed=seed)

        train_dataset = data_split["train"]
        test_dataset = data_split["test"]

    train_dataset = train_dataset.map(process_example)
    test_dataset = test_dataset.map(process_example)

    return train_dataset, test_dataset


def get_model(model_id, quantization_config=None, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        **kwargs,
    )
    return model


def get_peft_config(
    lora_target_modules: list[str] | None,
    r: int = 16,
    lora_alpha: int = 32,
    dropout: float = 0.05,
    lora_bias: bool = False,
    bias: Literal["none", "all", "lora_only"] = "none",
):
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        lora_bias=lora_bias,
        bias=bias,
        target_modules=lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    return peft_config


def get_quantization_config(
    load_in_4bit: bool = True,
    quant_type: Literal["nf4", "fp4"] = "nf4",
    use_double_quant: bool = True,
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16",
):
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[compute_dtype]

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=dtype,
    )


def get_sft_config(args: Args) -> SFTConfig:
    return SFTConfig(
        output_dir=str(args.output_dir),
        # Training
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        # Evaluation
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        # Checkpointing
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        # Memory optimization
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SFT
        max_length=args.max_seq_length,
        packing=False,
        # Logging and reproducibility
        seed=args.seed,
        report_to=args.report_to,
        # Evaluate and save on matching intervals
        load_best_model_at_end=(
            args.eval_strategy == "steps" and args.save_strategy == "steps"
        ),
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )


def run(
    model,
    tokenizer,
    train_dataset,
    test_dataset,
    args: Args,
):
    training_args = get_sft_config(args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "final_adapter"

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = trainer.evaluate()
    trainer.log_metrics("final_eval", metrics)
    trainer.save_metrics("final_eval", metrics)

    return trainer


def main(args: Args):
    configure_wandb(args)

    model_id = args.model_id

    tokenizer_id = args.tokenizer_id or model_id
    tokenizer = load_tokenizer(tokenizer_id)

    train_dataset, test_dataset = load_datasets(
        seed=args.seed,
        dataset_id=args.dataset_id,
        train_split=args.train_split,
        test_split=args.test_split,
        test_train_ratio=args.test_train_ratio,
        dataset_kwargs=args.dataset_kwargs,
    )

    peft_config = get_peft_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_bias=args.lora_bias,
        dropout=args.lora_dropout,
        bias=args.bias,
        lora_target_modules=args.lora_target_modules,
    )

    quantization_config = get_quantization_config(
        load_in_4bit=args.load_in_4bit,
        quant_type=args.bnb_4bit_quant_type,
        use_double_quant=args.bnb_4bit_use_double_quant,
        compute_dtype=args.bnb_4bit_compute_dtype,
    )

    model = get_model(model_id=model_id, quantization_config=quantization_config)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.config.use_cache = False

    trainer = run(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        args=args,
    )

    return trainer


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    hf_auth()

    args = tyro.cli(Args)

    main(args)
