from pathlib import Path

import torch
import warnings
import tiktoken

from .model import LLama3_xs

# from .tokenizer import load_tokenizer
from .config import ModelConfig


def load_gpt2_tokenizer():
    get2_tokenizer = tiktoken.get_encoding("gpt2")
    return get2_tokenizer


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

    tokenizer = load_gpt2_tokenizer()
    vocab_size = tokenizer.n_vocab

    checkpoint_path = Path("checkpoints/Llama32_180M.pt")

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

    texts = [
        "Once upon a time there was a pumpkin.",
        "A little girl went to the woods",
        "A boy told his sister a bedtime story about a flying cat",
        "The kids sat in a circle while Uncle narrated a story about a brave knight",
        "Dad was telling the kids an adventure tale about a pirate ship",
    ]
    for i, text in enumerate(texts):
        print(f"{i + 1:2d} input sentence {i}")
        tokenized = tokenizer.encode_ordinary(
            text,
        )
        token_ids = tokenized
        y = generate(token_ids, model, device, eos_id=tokenizer.eot_token)
        generated = tokenizer.decode(y.squeeze().tolist())

        print(generated)
        print(f"\n{'-' * 64}\n")
