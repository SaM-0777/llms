import argparse

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--text", type=str)
    return parser.parse_args()


def main(tokenizer_name: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        use_fast=True,
    )
    token_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    for token_id, token in zip(token_ids, tokens):
        print(f"{token_id:6} {token!r}")


if __name__ == "__main__":
    args = parse_args()

    if not args.model_name:
        raise ValueError(f"--model-name is not provided. Provide a valid model name.")
    if not args.text:
        args.text = """
def update(x):
    x -= 1
    x **= 2
    x //= 3
    x <<= 1
    x @= y
    return x
        """
    
    main(args.model_name, text=args.text)
