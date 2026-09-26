import os
from datetime import datetime
from itertools import cycle

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb

from .config import ModelConfig
from .dataset import MemmapDataset
from .model import LLama3_xs
from .trainer import Trainer
from .train_utils import get_model_stats, load_checkpoint, get_device_settings


def load_tokenizer(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
    )
    return tokenizer


def main(cfg: ModelConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    writer = SummaryWriter(log_dir=f"runs/llama32_{timestamp}")
    torch.manual_seed(cfg.seed)
    device, dtype, ptdtype, ctx, scaler = get_device_settings()

    cfg.device = torch.device(device)
    cfg.dtype = ptdtype

    tokenizer = load_tokenizer(cfg.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    cfg.vocab_size = vocab_size

    if cfg.wandb_log:
        wandb.init(
            project=cfg.project_name,
            name=f"llama32_{timestamp}",
            sync_tensorboard=True,
        )

    model = LLama3_xs(cfg)

    if cfg.resume_from:
        if not load_checkpoint(model, cfg.resume_from, device):
            return

    model.to(device)

    stats = get_model_stats(model, cfg)
    for name, value in stats.items():
        writer.add_scalar(
            name,
            value,
            0,
        )
    writer.flush()

    # compile the model
    print(f"Compiling the model...")
    model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),  # type: ignore
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        eps=1e-9,
        fused=(True if device == "cuda" else False),
    )

    # Scheduler
    num_optimizer_steps = (
        cfg.max_iters + cfg.gradient_accumulation_steps - 1
    ) // cfg.gradient_accumulation_steps
    warmup_optimizer_steps = cfg.warmup_steps

    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_optimizer_steps)
    scheduler_delay = CosineAnnealingLR(
        optimizer,
        T_max=(num_optimizer_steps - warmup_optimizer_steps),
        eta_min=cfg.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_delay],
        milestones=[warmup_optimizer_steps],
    )

    # Dataset loading
    num_workers = (
        cfg.num_dataset_workers
        if cfg.num_dataset_workers is not None
        else os.cpu_count() // 2
    )
    train_dataset = MemmapDataset(
        os.path.join(cfg.data_dir, "train.bin"), cfg.block_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,  # type: ignore
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_dataset = MemmapDataset(os.path.join(cfg.data_dir, "test.bin"), cfg.block_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,  # type: ignore
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,  # type: ignore
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
    )
    eval_iterators = {
        "train": cycle(train_eval_loader),
        "val": cycle(val_loader),
    }

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ctx=ctx,
        train_loader=train_loader,
        eval_iterators=eval_iterators,
        writer=writer,
        args=cfg,
        device=torch.device(device),
        timestamp=timestamp,
        max_norm=1.0,
    )

    train_loss, val_loss = trainer.train()

    if cfg.wandb_log and wandb.run is not None:
        wandb.finish()

    return train_loss, val_loss


if __name__ == "__main__":
    import tyro

    cfg = tyro.cli(ModelConfig)
    main(cfg)
