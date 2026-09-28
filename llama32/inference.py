from pathlib import Path

import torch
import warnings

from .model import LLama3_xs
from .tokenizer import load_tokenizer
from .config import ModelConfig


def generate(
    token_ids,
    model,
    device,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    eos_id: int | None = None,
):
    prompt = torch.tensor(token_ids, device=device).unsqueeze(dim=0)

    with torch.no_grad():
        y = model.generate(
            x=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )

    return y


if __name__ == "__main__":
    dtype = torch.bfloat16
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn(f"Using CPU...")
        device = "cpu"

    tokenizer = load_tokenizer("Qwen/Qwen3.8-27B")
    vocab_size = tokenizer.vocab_size

    checkpoint_path = Path("checkpoints/llama32_best_model_20260926_1527@190000.pt")

    cfg = ModelConfig()
    cfg.vocab_size = vocab_size
    model = LLama3_xs(cfg)

    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device(device), weights_only=True
    )

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
            state_dict[new_key] = value
        else:
            state_dict[key] = value

    model.load_state_dict(state_dict)
    model.to(device)
    model.to(dtype=dtype)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params / 1e6:.2f}M parameters")

    text = "Amazon.com (AMZN) will have a difficult time meeting analyst expectations this quarter given "
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = tokenized["input_ids"]

    y = generate(token_ids, model, device, eos_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(y.squeeze().tolist())

    print(generated)
    print(f"\n{'-' * 64}\n")
