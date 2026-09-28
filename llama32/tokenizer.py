from transformers import AutoTokenizer

def load_tokenizer(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
    )
    return tokenizer
