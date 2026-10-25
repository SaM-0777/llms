from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import random
from typing import Literal

import tyro
import torch
from datasets import load_dataset
from transformers import get_scheduler
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import rotate_checkpoints
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

import wandb
from tqdm import tqdm

from decider.dataset import encode_dataset
from decider.encoder import Encoded
from decider.loss import decision_loss
from decider.model import DecisionModel

LOG = logging.getLogger("DECIDER")


@dataclass
class Args:
    seed: int = 42
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    model_id: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "2b48083dfa97c3cbf0220cf1a5e6cffe0a511157"
    batch_size: int = 16
    dataset_id: str = "LocalLLaMA/typed-decisions"
    dataset_name: str | None = "all"
    dataset_streaming: bool = False
    train_split: str = "train"
    eval_split: str = "test"

    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: str = "all-linear"
    lora_bias: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"

    output_dir: str | Path = "output"
    resume_from_checkpoint: str | None = None
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 4
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
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool | None = None
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


def configure_wandb(args: Args) -> str | None:
    if args.report_to != "wandb":
        return None

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.project_name}_{timestamp}"

    config = {
        "seed": args.seed,
        "model_id": args.model_id,
        "tokenizer_id": args.model_id,
        "dataset_id": args.dataset_id,
        "train_split": args.train_split,
        "test_split": args.eval_split,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_bias": args.lora_bias,
        "bias": args.bias,
        "lora_target_modules": args.lora_target_modules,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "max_grad_norm": args.max_grad_norm,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_seq_length": args.max_seq_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        "torch_compile": args.torch_compile,
        "torch_compile_backend": args.torch_compile_backend,
        "torch_compile_mode": args.torch_compile_mode,
        "torch_compile_fullgraph": args.torch_compile_fullgraph,
        "torch_compile_dynamic": args.torch_compile_dynamic,
    }

    run = wandb.init(
        project=args.project_name,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags,
        config=config,
        job_type="sft",
        save_code=True,
        dir=str(output_dir),
        reinit=True,
    )

    LOG.info(
        "Initialized W&B: project=%s run=%s",
        args.project_name,
        run.name,
    )

    return run.id


def get_lora_config(
    lora_target_modules: str | list[str] | None,
    lora_r: int = 16,
    lora_alpha: int = 16,
    dropout: float = 0.0,
    lora_bias: bool = False,
    bias: Literal["none", "all", "lora_only"] = "none",
):
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        lora_bias=lora_bias,
        bias=bias,
        target_modules=lora_target_modules,
    )

    return lora_config


def get_model(args: Args):
    if args.lora_r < 1:
        raise ValueError("lora_r must be at least one")
    if args.lora_alpha < 1:
        raise ValueError("lora_alpha must be at least one")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative")

    model = DecisionModel(
        model_id=args.model_id,
        revision=args.revision,
        device=torch.device(args.device),
        train=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.backbone.requires_grad_(False)

    lora_config = get_lora_config(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        bias=args.bias,
        lora_target_modules=args.lora_target_modules,
    )
    model.backbone = get_peft_model(  # pyright: ignore[reportAttributeAccessIssue]
        model.backbone, lora_config
    )
    model.to(args.device)

    lora_parameters = []
    frozen_backbone_parameters = []

    for name, parameter in model.backbone.named_parameters():
        if "lora_" in name:
            if not parameter.requires_grad:
                raise RuntimeError(f"LoRA parameter is frozen: {name}")
            lora_parameters.append(parameter)
        else:
            if parameter.requires_grad:
                raise RuntimeError(f"Original backbone parameter is trainable: {name}")
            frozen_backbone_parameters.append(parameter)

    if not lora_parameters:
        raise RuntimeError("No LoRA parameters were created")
    if any("lora_" in name for name, _ in model.readout.named_parameters()):
        raise RuntimeError("LoRA was added to the readout")
    if not all(parameter.requires_grad for parameter in model.readout.parameters()):
        raise RuntimeError("Readout parameters must be trainable")

    model.readout.requires_grad_(True)
    if args.torch_compile:
        LOG.info(
            "Compiling model with torch.compile: backend=%s mode=%s fullgraph=%s dynamic=%s",
            args.torch_compile_backend,
            args.torch_compile_mode,
            args.torch_compile_fullgraph,
            args.torch_compile_dynamic,
        )
        model = torch.compile(
            model,
            backend=args.torch_compile_backend,
            mode=args.torch_compile_mode,
            fullgraph=args.torch_compile_fullgraph,
            dynamic=args.torch_compile_dynamic,
        )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    return model, optimizer


def save_checkpoint(
    accelerator: Accelerator,
    args: Args,
    trainer_state: TrainerState,
    *,
    global_step: int,
    final: bool = False,
) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = (
        output_dir / "final_checkpoint"
        if final
        else output_dir / f"checkpoint-{global_step}"
    )

    accelerator.wait_for_everyone()
    accelerator.save_state(
        output_dir=str(checkpoint_dir),
        safe_serialization=True,
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        trainer_state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
        if not final:
            rotate_checkpoints(
                output_dir=str(output_dir),
                save_total_limit=args.save_total_limit,
                checkpoint_prefix="checkpoint",
            )

    accelerator.wait_for_everyone()
    return checkpoint_dir


def evaluate(
    model: DecisionModel,
    eval_batches: list[list[Encoded]],
    args: Args,
) -> dict[str, float]:
    if not eval_batches:
        raise ValueError("Evaluation batches are empty")

    model.eval()
    total_loss = 0.0
    total_cross_entropy = 0.0
    total_brier = 0.0
    total_questions = 0
    correct_questions = 0

    eval_progress = tqdm(
        eval_batches,
        desc="Evaluate",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for encoded_batch in eval_progress:
            outputs = model(encoded_batch)
            components = decision_loss(
                outputs=outputs,
                chunks=encoded_batch,
                brier_weight=args.brier_weight,
            )

            if not torch.isfinite(components.total):
                raise RuntimeError("Nonfinite evaluation loss")

            batch_questions = sum(len(chunk.labels) for chunk in encoded_batch)
            if batch_questions == 0:
                raise ValueError("Evaluation batch produced no questions")

            total_loss += float(components.total) * batch_questions
            total_cross_entropy += float(components.cross_entropy) * batch_questions
            total_brier += float(components.brier) * batch_questions

            for chunk_outputs, chunk in zip(outputs, encoded_batch, strict=True):
                for logits, label in zip(chunk_outputs, chunk.labels, strict=True):
                    prediction = int(logits.argmax(dim=-1).item())
                    correct_questions += int(prediction == label)
                    total_questions += 1

            eval_progress.set_postfix(
                loss=total_loss / total_questions,
                acc=correct_questions / total_questions,
            )

    if total_questions == 0:
        raise ValueError("Evaluation produced no questions")

    model.train()
    return {
        "eval/loss": total_loss / total_questions,
        "eval/cross_entropy": total_cross_entropy / total_questions,
        "eval/brier": total_brier / total_questions,
        "eval/accuracy": correct_questions / total_questions,
        "eval/questions": float(total_questions),
    }


def train(
    model: DecisionModel,
    optimizer: torch.optim.Optimizer,
    train_batches: list[list[Encoded]],
    eval_batches: list[list[Encoded]],
    accelerator: Accelerator,
    args: Args,
) -> None:
    if not train_batches:
        raise ValueError("Training batches are empty")
    if args.num_train_epochs < 1:
        raise ValueError("num_train_epochs must be at least one")
    if int(args.num_train_epochs) != args.num_train_epochs:
        raise ValueError("num_train_epochs must be an integer for this loop")
    if args.logging_steps < 1:
        raise ValueError("logging_steps must be at least one")
    if args.eval_steps < 1:
        raise ValueError("eval_steps must be at least one")
    if args.save_steps < 1:
        raise ValueError("save_steps must be at least one")
    if args.save_total_limit < 1:
        raise ValueError("save_total_limit must be at least one")
    if not eval_batches:
        raise ValueError("Evaluation batches are empty")

    epochs = int(args.num_train_epochs)
    total_training_steps = len(train_batches) * epochs

    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if args.warmup_steps > total_training_steps:
        raise ValueError("warmup_steps cannot exceed total training steps")

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    trainer_state = TrainerState(
        epoch=0.0,
        global_step=0,
        max_steps=total_training_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=epochs,
    )

    if args.resume_from_checkpoint is not None:
        checkpoint_dir = Path(args.resume_from_checkpoint).expanduser().resolve()
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(f"Trainer state is missing: {trainer_state_path}")

        accelerator.load_state(input_dir=str(checkpoint_dir))
        trainer_state = TrainerState.load_from_json(str(trainer_state_path))
        global_step = trainer_state.global_step
        if global_step > total_training_steps:
            raise ValueError(
                "Checkpoint global_step is beyond the configured training run: "
                f"{global_step} > {total_training_steps}"
            )

    start_epoch = global_step // len(train_batches)
    resume_batch = global_step % len(train_batches)

    def log_evaluation(step: int, progress: tqdm | None = None) -> None:
        eval_metrics = evaluate(
            model=model,
            eval_batches=eval_batches,
            args=args,
        )

        print(eval_metrics)
        if args.report_to == "wandb":
            wandb.log(eval_metrics, step=step)

        if progress is not None:
            progress.set_postfix(
                eval_loss=eval_metrics["eval/loss"],
                eval_acc=eval_metrics["eval/accuracy"],
            )

    for epoch in range(start_epoch, epochs):
        loss = 0.0
        cross_entropy = 0.0
        brier = 0.0
        batches = 0

        epoch_batches = list(train_batches)
        random.Random(args.seed + epoch).shuffle(epoch_batches)
        batch_start = resume_batch if epoch == start_epoch else 0

        progress = tqdm(
            epoch_batches[batch_start:],
            total=len(epoch_batches),
            initial=batch_start,
            desc=f"Train epoch {epoch + 1}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_index, encoded_batch in enumerate(
            progress,
            start=batch_start,
        ):
            outputs = model(encoded_batch)
            components = decision_loss(
                outputs=outputs,
                chunks=encoded_batch,
                brier_weight=args.brier_weight,
            )

            if not torch.isfinite(components.total):
                raise RuntimeError("Nonfinite training loss")

            accelerator.backward(components.total)

            loss += float(components.total.detach())
            cross_entropy += float(components.cross_entropy.detach())
            brier += float(components.brier.detach())
            batches += 1

            if not torch.isfinite(components.total.detach()):
                raise RuntimeError(f"Nonfinite loss at step {global_step + 1}")

            gradient_norm: torch.Tensor | None = None
            if args.max_grad_norm:
                gradient_norm = accelerator.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.max_grad_norm,
                )
                if gradient_norm is None:
                    raise RuntimeError("Gradient norm is None")
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Nonfinite gradient norm")

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            trainer_state.global_step = global_step
            trainer_state.epoch = epoch + (batch_index + 1) / len(epoch_batches)

            if global_step % args.logging_steps == 0:
                metrics = {
                    "train/loss": loss / batches,
                    "train/cross_entropy": cross_entropy / batches,
                    "train/brier": brier / batches,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                }

                if gradient_norm:
                    metrics["train/gradient_norm"] = float(gradient_norm)

                print(metrics)

                if args.report_to == "wandb":
                    wandb.log(metrics, step=global_step)

                progress.set_postfix(
                    loss=metrics["train/loss"],
                    lr=metrics["train/learning_rate"],
                )

                loss = 0.0
                cross_entropy = 0.0
                brier = 0.0
                batches = 0

            if global_step % args.eval_steps == 0:
                log_evaluation(global_step, progress)

            if global_step % args.save_steps == 0:
                checkpoint_dir = save_checkpoint(
                    accelerator=accelerator,
                    args=args,
                    trainer_state=trainer_state,
                    global_step=global_step,
                )
                LOG.info("Saved checkpoint: %s", checkpoint_dir)

    if global_step % args.eval_steps != 0 or start_epoch == epochs:
        log_evaluation(global_step)

    trainer_state.global_step = global_step
    trainer_state.epoch = float(epochs)
    final_checkpoint = save_checkpoint(
        accelerator=accelerator,
        args=args,
        trainer_state=trainer_state,
        global_step=global_step,
        final=True,
    )
    LOG.info("Saved final checkpoint: %s", final_checkpoint)


def main(args: Args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator()

    # Model
    model, optimizer = get_model(args)
    encoder = model.encoder

    # Dataset
    dataset_id = args.dataset_id
    dataset_name = args.dataset_name

    train_dataset = load_dataset(dataset_id, dataset_name, split=args.train_split)
    eval_dataset = load_dataset(dataset_id, dataset_name, split=args.eval_split)

    train_rows = encode_dataset(train_dataset, encoder=encoder)
    eval_rows = encode_dataset(eval_dataset, encoder=encoder)

    configure_wandb(args)
    train(
        model=model,
        optimizer=optimizer,
        train_batches=train_rows,
        eval_batches=eval_rows,
        accelerator=accelerator,
        args=args,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    main(args)
