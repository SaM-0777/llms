from datetime import datetime
from pathlib import Path
import logging
import random
from typing import Any, Literal, cast

import tyro
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

import wandb

from decider.config import Args
from decider.dataset import (
    EncodedBatchDataset,
    collate_encoded_batches,
    encode_dataset,
)
from decider.encoder import Encoded
from decider.loss import decision_loss
from decider.model import DecisionModel

LOG = logging.getLogger("DECIDER")


class DeciderTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        brier_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._train_question_count = 0
        self._train_cross_entropy = 0.0
        self._train_brier = 0.0
        self._eval_totals: list[float] | None = None
        self._brier_weight = brier_weight

    @staticmethod
    def _question_count(chunks: list[Encoded]) -> int:
        question_count = sum(len(chunk.labels) for chunk in chunks)
        if question_count == 0:
            raise ValueError("Encoded batch produced no questions")
        return question_count

    @staticmethod
    def _accuracy(outputs: list[list[torch.Tensor]], chunks: list[Encoded]) -> int:
        correct = 0
        for chunk_outputs, chunk in zip(outputs, chunks, strict=True):
            for logits, label in zip(chunk_outputs, chunk.labels, strict=True):
                correct += int(int(logits.argmax(dim=-1).item()) == label)
        return correct

    def _compute_components(
        self,
        model: torch.nn.Module,
        chunks: list[Encoded],
    ):
        outputs = model(chunks)
        components = decision_loss(
            outputs=outputs,
            chunks=chunks,
            brier_weight=self._brier_weight,
        )
        if not torch.isfinite(components.total):
            raise RuntimeError("Nonfinite loss")
        return outputs, components

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, list[Encoded] | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        chunks = inputs["chunks"]
        outputs, components = self._compute_components(model, chunks)
        question_count = self._question_count(chunks)

        if model.training:
            self._train_question_count += question_count
            self._train_cross_entropy += (
                float(components.cross_entropy.detach()) * question_count
            )
            self._train_brier += float(components.brier.detach()) * question_count

        if return_outputs:
            return components.total, outputs
        return components.total

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        logs = dict(logs)
        if "loss" in logs and self._train_question_count:
            logs["train/loss"] = logs["loss"]
            logs["train/cross_entropy"] = (
                self._train_cross_entropy / self._train_question_count
            )
            logs["train/brier"] = self._train_brier / self._train_question_count
            if "learning_rate" in logs:
                logs["train/learning_rate"] = logs["learning_rate"]
            self._train_question_count = 0
            self._train_cross_entropy = 0.0
            self._train_brier = 0.0
        super().log(logs, start_time=start_time)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, list[Encoded] | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        inputs = self._prepare_inputs(inputs)
        chunks = inputs["chunks"]
        with torch.no_grad():
            outputs, components = self._compute_components(model, chunks)

        question_count = self._question_count(chunks)
        correct = self._accuracy(outputs, chunks)
        totals = torch.tensor(
            [
                float(components.total.detach()) * question_count,
                float(components.cross_entropy.detach()) * question_count,
                float(components.brier.detach()) * question_count,
                float(correct),
                float(question_count),
            ],
            device=components.total.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_totals = cast(
            torch.Tensor,
            self.accelerator.gather_for_metrics(totals),
        )
        totals = gathered_totals.sum(dim=0)
        if self._eval_totals is None:
            self._eval_totals = [0.0] * 5
        for index, value in enumerate(totals.tolist()):
            self._eval_totals[index] += value

        return components.total.detach(), None, None

    def evaluation_loop(self, *args: Any, **kwargs: Any):
        self._eval_totals = None
        metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
        output = super().evaluation_loop(*args, **kwargs)
        if self._eval_totals is None or self._eval_totals[4] == 0:
            raise ValueError("Evaluation produced no questions")

        eval_totals = cast(
            list[float] | None,
            self._eval_totals,
        )
        if eval_totals is None:
            raise ValueError("Evaluation produced no question totals")
        if eval_totals[4] == 0:
            raise ValueError("Evaluation produced no questions")

        total_loss, cross_entropy, brier, correct, questions = eval_totals

        metrics = dict(output.metrics or {})
        metrics.update(
            {
                f"{metric_key_prefix}_loss": total_loss / questions,
                f"{metric_key_prefix}_cross_entropy": cross_entropy / questions,
                f"{metric_key_prefix}_brier": brier / questions,
                f"{metric_key_prefix}_accuracy": correct / questions,
                f"{metric_key_prefix}_questions": questions,
            }
        )
        return output._replace(metrics=metrics)


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
        "full_finetuning": args.full_finetuning,
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
        dtype=torch.bfloat16,
        device=torch.device(args.device),
        train=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.full_finetuning:
        model.backbone.requires_grad_(True)
    else:
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

    if args.full_finetuning:
        if not all(
            parameter.requires_grad for parameter in model.backbone.parameters()
        ):
            raise RuntimeError(
                "All backbone parameters must be trainable for full fine-tuning"
            )
    else:
        lora_parameters = []
        for name, parameter in model.backbone.named_parameters():
            if "lora_" in name:
                if not parameter.requires_grad:
                    raise RuntimeError(f"LoRA parameter is frozen: {name}")
                lora_parameters.append(parameter)
            elif parameter.requires_grad:
                raise RuntimeError(f"Original backbone parameter is trainable: {name}")

        if not lora_parameters:
            raise RuntimeError("No LoRA parameters were created")

    if any("lora_" in name for name, _ in model.readout.named_parameters()):
        raise RuntimeError("LoRA was added to the readout")
    if not all(parameter.requires_grad for parameter in model.readout.parameters()):
        raise RuntimeError("Readout parameters must be trainable")

    model.readout.requires_grad_(True)
    return model


def train(
    model: DecisionModel,
    train_batches: list[list[Encoded]],
    eval_batches: list[list[Encoded]],
    args: Args,
) -> None:
    if not train_batches:
        raise ValueError("Training batches are empty")
    if not eval_batches:
        raise ValueError("Evaluation batches are empty")
    if args.num_train_epochs < 1:
        raise ValueError("num_train_epochs must be at least one")
    if args.logging_steps < 1:
        raise ValueError("logging_steps must be at least one")
    if args.eval_steps < 1:
        raise ValueError("eval_steps must be at least one")
    if args.save_steps < 1:
        raise ValueError("save_steps must be at least one")
    if args.save_total_limit < 1:
        raise ValueError("save_total_limit must be at least one")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=args.seed,
        use_cpu=args.device == "cpu",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        max_grad_norm=args.max_grad_norm or 1.0,
        bf16=True,
        bf16_full_eval=True,
        torch_compile=args.torch_compile,
        torch_compile_backend=(
            args.torch_compile_backend if args.torch_compile else None
        ),
        torch_compile_mode=args.torch_compile_mode if args.torch_compile else None,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = DeciderTrainer(
        model=model,
        args=training_args,
        train_dataset=EncodedBatchDataset(train_batches),
        eval_dataset=EncodedBatchDataset(eval_batches),
        data_collator=collate_encoded_batches,
        processing_class=model.tokenizer,
        brier_weight=args.brier_weight,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_checkpoint = output_dir / "final_checkpoint"
    trainer.save_model(str(final_checkpoint))
    LOG.info("Saved final model: %s", final_checkpoint)


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

    # Model
    model = get_model(args)
    encoder = model.encoder

    # Dataset
    dataset_id = args.dataset_id
    dataset_name = args.dataset_name

    train_dataset = load_dataset(dataset_id, dataset_name, split=args.train_split)
    eval_dataset = load_dataset(dataset_id, dataset_name, split=args.eval_split)

    train_rows = encode_dataset(
        train_dataset,
        encoder=encoder,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
    )
    eval_rows = encode_dataset(
        eval_dataset,
        encoder=encoder,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
    )

    configure_wandb(args)
    train(
        model=model,
        train_batches=train_rows,
        eval_batches=eval_rows,
        args=args,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    main(args)
