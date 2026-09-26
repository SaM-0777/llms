import math
import os

import torch
from contextlib import nullcontext


def load_checkpoint(model, resume_from_path, device):
    if os.path.exists(resume_from_path):
        print(f"Loading weights from checkpoint: {resume_from_path}")
        checkpoint = torch.load(
            resume_from_path, map_location=device, weights_only=True
        )

        state_dict = {}
        for key, value in checkpoint.model.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                state_dict[new_key] = value
            else:
                state_dict[key] = value

        model.load_state_dict(state_dict)
        return True
    else:
        print(f"Checkpoint file not found: {resume_from_path}")
        return False


def get_device_settings():
    """Determines the optimal device and data type for training."""
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = "bfloat16"
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        device = "cpu"
        dtype = "bfloat16"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    print(f"Using device: {device}")
    print(f"dtype: {dtype}")

    ctx = (
        torch.amp.autocast(device_type=device, dtype=ptdtype)
        if device != "cpu"
        else nullcontext()
    )

    # GradScaler is only needed for float16
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    return device, dtype, ptdtype, ctx, scaler


def get_model_stats(model, cfg):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer_steps = math.ceil(cfg.max_iters / cfg.gradient_accumulation_steps)
    tokens_per_microstep = cfg.batch_size * cfg.block_size
    tokens_per_optimizer_step = tokens_per_microstep * cfg.gradient_accumulation_steps

    total_tokens = cfg.max_iters * tokens_per_microstep

    return {
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/embedding_dim": cfg.embedding_dim,
        "model/hidden_dim": cfg.hidden_dim,
        "model/num_blocks": cfg.num_blocks,
        "model/num_heads": cfg.num_heads,
        "model/num_kv_heads": cfg.num_kv_heads,
        "model/head_dim": cfg.head_dim,
        "model/vocab_size": cfg.vocab_size,
        "model/max_sequence_length": cfg.max_sequence_length,
        "train/batch_size": cfg.batch_size,
        "train/block_size": cfg.block_size,
        "train/gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "train/optimizer_steps": optimizer_steps,
        "train/microsteps": cfg.max_iters,
        "train/tokens_per_microstep": tokens_per_microstep,
        "train/tokens_per_optimizer_step": tokens_per_optimizer_step,
        "train/total_tokens": total_tokens,
        "train/learning_rate": cfg.learning_rate,
        "train/min_lr": cfg.min_lr,
        #"train/warmup_optimizer_steps": cfg.warmup_optimizer_steps,
        "Train/warmup_steps": cfg.warmup_steps,
        "train/weight_decay": cfg.weight_decay,
    }
