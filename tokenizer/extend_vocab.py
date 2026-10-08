import argparse
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--tokens-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="extended_tokenizer")
    return parser.parse_args()


def main(model_name: str, new_tokens: list[str], output_dir: str | Path):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    original_vocab_size = len(tokenizer)
    existing_tokens = []
    missing_tokens = []

    for token in new_tokens:
        if token in tokenizer.get_vocab():
            existing_tokens.append(token)
        else:
            missing_tokens.append(token)

    print()
    print("=" * 60)
    print("Python Token Analysis")
    print("=" * 60)
    print(f"Original vocabulary: {original_vocab_size:,}")
    print(f"New tokens:       {len(new_tokens):,}")
    print(f"Already present:     {len(existing_tokens):,}")
    print(f"Missing:             {len(missing_tokens):,}")

    print()
    print("Missing tokens:")
    print("-" * 60)

    for token in missing_tokens:
        print(repr(token))

    added = tokenizer.add_tokens(missing_tokens)
    final_vocab_size = len(tokenizer)

    print()
    print("=" * 60)
    print("Tokenizer Extension")
    print("=" * 60)
    print(f"Tokens added:        {added:,}")
    print(f"Final vocabulary:    {final_vocab_size:,}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)

    print()
    print(f"Saved tokenizer to: {output_path}")

    print()
    print("=" * 60)
    print("Validation")
    print("=" * 60)

    for token in missing_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        single_token = len(encoded) == 1 and encoded[0] == token_id

        print(f"{token!r:20} " f"id={token_id:<7} " f"single_token={single_token}")


if __name__ == "__main__":
    args = parse_args()

    if not args.model_name:
        raise ValueError(f"--model-name is not provided. Provide a valid model name.")
    if not args.tokens_dir:
        raise ValueError(
            f"--tokens-dir is not provided. Provide a valid tokens dir .txt file. Make sure each line contain a single token string"
        )

    tokens_file_path = Path(args.tokens_dir)
    if tokens_file_path.exists() and tokens_file_path.is_file():
        with open(tokens_file_path, "r", encoding="utf-8") as f:
            tokens = f.read().splitlines()
        print(f"Found {len(tokens)} tokens.")
    else:
        raise ValueError(f"Provided tokens path either doesn't exist or is not a file.")

    main(model_name=args.model_name, new_tokens=tokens, output_dir=args.output_dir)
