import gigatoken as gt
from transformers import AutoTokenizer

hf_tokenizer = AutoTokenizer.from_pretrained(
    "tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended_II",
    use_fast=True,
    trust_remote_code=False,
)

tokenizer = gt.Tokenizer(hf_tokenizer).as_hf()

texts = [
    "def add(a, b): return a + b",
    "The derivative of x² is 2x.",
]
encoded = tokenizer(texts, add_special_tokens=False)

for text, tokens in zip(texts, encoded["input_ids"]):
    print(f"Text: {text!r}")
    print(f"Tokens ({len(tokens)}): {tokens}")
