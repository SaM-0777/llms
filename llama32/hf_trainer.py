import os
import math
import time

from typing import Any
from datasets.arrow_dataset import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer as HFTrainer


class Trainer(HFTrainer):
    def __init__(self, *args, min_lr: float, **kwargs):
        super().__init__(*args, **kwargs)

        self.min_lr = min_lr
        self._training_start_time = None
        self._last_log_time = time.time()
        self._last_tokens_seen = 0

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError(f"Training requires a train_dataset.")

        dataloader = DataLoader(
            self.train_dataset,
            collate_fn=self._collate_batch,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=(
                self.args.dataloader_persistent_workers
                if self.args.dataloader_num_workers > 0
                else False
            ),
        )

        return self.accelerator.prepare(dataloader)

    def get_eval_dataloader(
        self, eval_dataset: str | Dataset | None = None
    ) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError(f"Evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        dataloader = DataLoader(
            eval_dataset,
            collate_fn=self._collate_batch,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=(
                self.args.dataloader_persistent_workers
                if self.args.dataloader_num_workers > 0
                else False
            ),
        )

        return self.accelerator.prepare(dataloader)

    def _collate_batch(self, features):
        return {
            "x": torch.stack([torch.as_tensor(feature["x"]) for feature in features]),
            "targets": torch.stack(
                [torch.as_tensor(feature["targets"]) for feature in features]
            ),
        }

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        x = inputs["x"]
        targets = inputs["targets"]
        logits, loss = model(
            x=x,
            targets=targets,
        )

        if loss is None:
            raise RuntimeError("Model returned None for loss.")
        if return_outputs:
            return loss, logits

        return loss

    def create_optimizer(self, model: nn.Module | None = None) -> optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer

        if model is None:
            model = self.model

        assert model is not None

        decay_parameters = self.get_decay_parameter_names(model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.weight_decay,
            eps=self.args.adam_epsilon,
            fused=True if self.args.device == "cuda" else None,
        )
        self.optimizer = optimizer
        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: optim.Optimizer | None = None
    ) -> LRScheduler:
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        if optimizer is None:
            optimizer = self.optimizer
        assert optimizer is not None, f"Optimizer must exist before scheduler."

        warmup_steps = int(self.args.warmup_steps)
        if warmup_steps >= num_training_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be "
                f"less than total training steps "
                f"({num_training_steps})."
            )

        scheduler_warmup = LinearLR(
            optimizer,
            total_iters=warmup_steps,
        )
        scheduler_delay = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=self.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_delay],
            milestones=[warmup_steps],
        )
        self.lr_scheduler = scheduler
        self._created_lr_scheduler = True

        return self.lr_scheduler

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        now = time.time()

        if self._training_start_time is None:
            self._training_start_time = now

        if self._last_log_time is not None:
            elapsed = now - self._last_log_time
            tokens_seen = self.state.num_input_tokens_seen
            tokens_delta = tokens_seen - self._last_tokens_seen

            if elapsed > 0:
                logs["tokens_per_second"] = tokens_delta / elapsed

            logs["tokens_seen"] = float(tokens_seen)
            self._last_tokens_seen = tokens_seen

        self._last_log_time = now
        super().log(
            logs,
            start_time=start_time,
        )

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        loss_key = f"{metric_key_prefix}_loss"

        if loss_key in metrics:
            loss = metrics[loss_key]
            try:
                metrics[f"{metric_key_prefix}_perplexity"] = math.exp(loss)
            except OverflowError:
                metrics[f"{metric_key_prefix}_perplexity"] = float("inf")

        return metrics
