import torch
from transformers import AutoTokenizer

from qwen38.config import Config
from qwen38.model import Qwen38
from qwen38.utils import count_parameters

cfg = Config()

tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
vocab_size = tokenizer.vocab_size
cfg.vocab_size = vocab_size

cfg.device = torch.device("cpu")
model = Qwen38(cfg=cfg).to(cfg.device)

params = count_parameters(model)
# 2.68B
print(f"Trainable {params["trainable_params"]}, Total {params["total_params"]}")
