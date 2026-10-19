import argparse

from transformers import AutoTokenizer


MODEL_ID = "tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended_II"

CONVERSATION = """Hello how are you
I am fine def definition
"""


def main():
    parser = argparse.ArgumentParser(
        description="Validate Nemotron tokenization and lexical boundaries."
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--base-model")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print("\nTokenizer loaded successfully.")
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")

    # Inspect the original pre-tokenizer when the fast backend is available.
    backend = getattr(tokenizer, "backend_tokenizer", None)

    if backend is not None:
        print(f"\nOriginal pre-tokenizer:\n{backend.pre_tokenizer}")

    # Encode the conversation using the unmodified tokenizer.
    token_ids = tokenizer.encode(
        CONVERSATION,
        add_special_tokens=False,
    )

    token_strings = tokenizer.convert_ids_to_tokens(token_ids)

    print("\n" + "=" * 80)
    print("INPUT TEXT")
    print("=" * 80)
    print(CONVERSATION)

    print("\n" + "=" * 80)
    print(f"TOKENIZATION ({len(token_ids)} tokens)")
    print("=" * 80)

    for index, (token_id, token_string) in enumerate(
        zip(token_ids, token_strings)
    ):
        print(
            f"{index:4d} | "
            f"ID: {token_id:7d} | "
            f"Token: {token_string!r}"
        )

    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    print("\n" + "=" * 80)
    print("ROUND-TRIP VALIDATION")
    print("=" * 80)
    print(f"Original and decoded text match: {decoded == CONVERSATION}")

    if decoded != CONVERSATION:
        print("\nDecoded text:")
        print(repr(decoded))

    # Check how individual lexical strings are represented.
    probes = [
        "def",
        "!=",
        "<<=",
        "nonlocal",
        "\\alpha",
        "∇",
        "calculate_total",
    ]

    print("\n" + "=" * 80)
    print("LEXICAL TOKEN PROBES")
    print("=" * 80)

    for text in probes:
        ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = tokenizer.convert_ids_to_tokens(ids)

        print(f"{text!r}")
        print(f"  IDs:    {ids}")
        print(f"  Pieces: {pieces}")
        print(f"  Single token: {len(ids) == 1}")

    if args.base_model:
        print("\n" + "=" * 80)
        print("BOUNDARY REGRESSION CHECKS")
        print("=" * 80)

        base = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
        )
        base_vocab = base.get_vocab()
        extended_vocab = tokenizer.get_vocab()
        ids_preserved = all(
            extended_vocab.get(token) == token_id
            for token, token_id in base_vocab.items()
        )
        cases = {
            "for item in format": "for",
            "a <<= 2": "<<=",
            r"\alpha + \beta": r"\alpha",
            r"\alphabet": None,
            "Hello there, how are you today?": None,
        }

        print(f"Existing base vocabulary IDs preserved: {ids_preserved}")
        for text, atomic_token in cases.items():
            base_ids = base.encode(text, add_special_tokens=False)
            extended_ids = tokenizer.encode(text, add_special_tokens=False)
            base_pieces = base.convert_ids_to_tokens(base_ids)
            extended_pieces = tokenizer.convert_ids_to_tokens(extended_ids)
            passed = (
                atomic_token in extended_pieces
                if atomic_token is not None
                else base_pieces == extended_pieces
            )
            print(f"{text!r}: {'PASS' if passed else 'FAIL'}")
            print(f"  base:     {base_pieces}")
            print(f"  extended: {extended_pieces}")


if __name__ == "__main__":
    main()
