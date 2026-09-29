import os

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .train import load_gpt2_tokenizer
from .model import LLama3_xs
from .dataset import MemmapDataset


def main():
    cfg = ModelConfig()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    tokenizer = load_gpt2_tokenizer()
    vocab_size = tokenizer.n_vocab
    cfg.vocab_size = vocab_size

    print(f"Device:           {device}")
    print(f"Dtype:            {cfg.dtype}")
    print(f"Batch size:       {cfg.batch_size}")
    print(f"Block size:       {cfg.block_size}")
    print(f"Embedding dim:    {cfg.embedding_dim}")
    print(f"Hidden dim:       {cfg.hidden_dim}")
    print(f"Num blocks:       {cfg.num_blocks}")
    print(f"Num heads:        {cfg.num_heads}")
    print(f"Num KV heads:     {cfg.num_kv_heads}")
    print(f"Head dim:         {cfg.head_dim}")
    print(f"Vocab size:       {cfg.vocab_size}")

    train_dataset = MemmapDataset(
        os.path.join(cfg.data_dir, "gpt2_test.bin"), cfg.block_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,  # type: ignore
        pin_memory=True if device == "cuda" else False,
        persistent_workers=False,
    )
    inputs, targets = next(iter(train_loader))
    print(f"inputs.shape:     {inputs.shape}")
    print(f"targets.shape:    {targets.shape}")
    print(f"inputs.dtype:     {inputs.dtype}")
    print(f"targets.dtype:    {targets.dtype}")

    inputs = inputs.to(device)
    targets = targets.to(device)

    model = LLama3_xs(cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    with torch.no_grad():
        logits, loss = model(
            inputs,
            targets=targets,
            start_pos=0,
        )

    print(f"logits.shape:     {logits.shape}")
    print(f"logits.dtype:     {logits.dtype}")
    print(f"logits.device:    {logits.device}")

    if loss is not None:
        print(f"loss:             {loss.item():.6f}")
        print(f"loss dtype:       {loss.dtype}")


if __name__ == "__main__":
    main()


# import torch

# from .rope import (
#    compute_rope_params,
#    apply_rotary_emb,
#    apply_polar_rotary_emb,
#    compute_polar_rope_params,
# )

# torch.manual_seed(42)

# B = 2
# S = 16
# H = 8
# H_KV = 4
# D = 64

# q = torch.randn(B, S, H, D)
# k = torch.randn(B, S, H_KV, D)

## New real-valued frequencies
# cos, sin = compute_rope_params(
#    dim=D,
#    context_length=S,
# )

## Old complex frequencies
# freq_cis = compute_polar_rope_params(
#    dim=D,
#    context_length=S,
# )

# q_real, k_real = apply_rotary_emb(
#    q,
#    k,
#    (cos, sin),
# )

# q_complex, k_complex = apply_polar_rotary_emb(
#    q,
#    k,
#    freq_cis,
# )

# print("q max error:")
# print((q_real - q_complex).abs().max())

# print("k max error:")
# print((k_real - k_complex).abs().max())

# print(
#    "q allclose:",
#    torch.allclose(q_real, q_complex, atol=1e-6, rtol=1e-5),
# )

# print(
#    "k allclose:",
#    torch.allclose(k_real, k_complex, atol=1e-6, rtol=1e-5),
# )
