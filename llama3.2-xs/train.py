import os
from datetime import datetime
from itertools import cycle
import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .config import ModelConfig
from .dataset import MemmapDataset
from .model import LLama3_xs
from .trainer import Trainer
from .train_utils import load_checkpoint, get_device_settings


def main(cfg: ModelConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    writer = SummaryWriter(log_dir=f"runs/llama3_xs_{timestamp}")
    torch.manual_seed(cfg.seed)
    device, dtype, ptdtype, ctx, scaler = get_device_settings()

    cfg.device = torch.device(device)
    cfg.dtype = ptdtype

    model = LLama3_xs(cfg)

    if cfg.resume_from:
        if not load_checkpoint(model, cfg.resume_from, device):
            return

    model.to(device)

    # compile the model
    print(f"Compiling the model...")
    model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-9,
        fused=(True if device == "cuda" else False),
    )

    # Scheduler
    num_optimizer_steps = cfg.max_iters // cfg.gradient_accumulation_step
    warmup_optimizer_steps = cfg.warmup_steps // cfg.gradient_accumulation_step

    scheduler_warmup = LinearLR(optimizer, total_iters=cfg.warmup_steps)
    scheduler_delay = CosineAnnealingLR(
        optimizer,
        T_max=(num_optimizer_steps - warmup_optimizer_steps),
        eta_min=cfg.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_delay],
        milestones=[cfg.warmup_steps],
    )

    # Dataset loading
    train_dataset = MemmapDataset(
        os.path.join(cfg.data_dir, "train.bin"), cfg.block_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,  # type: ignore
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
    )

    val_dataset = MemmapDataset(os.path.join(cfg.data_dir, "test.bin"), cfg.block_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=os.cpu_count() // 2,  # type: ignore
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
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
    
