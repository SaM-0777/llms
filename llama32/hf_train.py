import os
from datetime import datetime

import tiktoken
import torch
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from wandb.util import np

from .config import ModelConfig
from .dataset import MemmapDataset
from .model import LLama3_xs
from .hf_trainer import Trainer
from .train_utils import get_device_settings, get_model_stats


def load_gpt2_tokenizer():
    get2_tokenizer = tiktoken.get_encoding("gpt2")
    return get2_tokenizer


def compute_metrics(eval_prediction):
    logits = eval_prediction.predictions
    targets = eval_prediction.label_ids
    predictions = np.argmax(logits, axis=-1)
    correct = (predictions == targets).sum()
    total = targets.size
    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
    }


def main(cfg: ModelConfig):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    torch.manual_seed(cfg.seed)
    device, dtype, ptdtype, ctx, scaler = get_device_settings()

    cfg.device = torch.device(device)
    cfg.dtype = ptdtype

    # tokenizer = load_tokenizer(cfg.tokenizer_name)
    tokenizer = load_gpt2_tokenizer()
    vocab_size = tokenizer.n_vocab
    cfg.vocab_size = vocab_size

    cfg.use_kv_cache = False  # do not use kv_cache in this training setup
    model = LLama3_xs(cfg)
    model.to(device)

    stats = get_model_stats(model, cfg)
    print(f"Trainable model params {stats.get("model/trainable_params")}")
    print(f"Total model params {stats.get("model/total_params")}")

    train_dataset = MemmapDataset(
        os.path.join(cfg.data_dir, "gpt2_train.bin"), cfg.block_size
    )
    eval_dataset = MemmapDataset(
        os.path.join(cfg.data_dir, "gpt2_test.bin"), cfg.block_size
    )

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        # trainer
        max_steps=cfg.max_iters,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_grad_norm=cfg.max_grad_norm,
        gradient_checkpointing=cfg.gradient_checkpointing,
        # optimizer
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        optim="adamw_torch_fused",
        # eval
        eval_strategy="steps",
        eval_steps=cfg.eval_intervals,
        # utils
        bf16=True,
        torch_compile=cfg.torch_compile,
        # log
        logging_strategy="steps",
        logging_steps=cfg.log_interval,
        # save
        save_strategy="steps",
        save_steps=cfg.save_interval,
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        # dataloader
        dataloader_num_workers=(
            cfg.num_dataset_workers if cfg.num_dataset_workers is not None else os.cpu_count() // 2  # type: ignore
        ),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=(
            cfg.num_dataset_workers > 0
            if cfg.num_dataset_workers is not None
            else False
        ),
        # token accounting
        include_num_input_tokens_seen="all",
        # reporting
        report_to=(["wandb", "tensorboard"] if cfg.wandb_log else ["tensorboard"]),
        run_name=f"llama32_{timestamp}",
        # hub
        push_to_hub=False,
    )

    trainer = Trainer(
        model,
        min_lr=cfg.min_lr,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    last_checkpoint = get_last_checkpoint(cfg.output_dir)
    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: " f"{last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    trainer.save_model(
        os.path.join(
            cfg.output_dir,
            "final",
        )
    )


if __name__ == "__main__":
    import tyro

    # python -m llama32.train --max_iters 500000 --eval_intervals 10000 --wandb_log
    cfg = tyro.cli(ModelConfig)
    main(cfg)
