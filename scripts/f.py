import keyword
import tokenize as py_tokenize
from pathlib import Path

from transformers import AutoTokenizer
from tokenizers import Regex
from tokenizers import pre_tokenizers


INPUT_DIR = Path(
    "tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-py-extended"
)

OUTPUT_DIR = Path(
    "tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-py-lexical"
)


def get_python_tokens():
    keywords = list(keyword.kwlist) + list(keyword.softkwlist)
    operators = list(py_tokenize.EXACT_TOKEN_TYPES.keys())

    return list(dict.fromkeys(keywords + operators))


def build_python_pattern(tokens):
    keywords = [
        token
        for token in tokens
        if token.isidentifier()
    ]

    operators = [
        token
        for token in tokens
        if not token.isidentifier()
    ]

    # Longest operators first:
    # **= before ** before *
    # >>= before >> before >
    operators.sort(key=len, reverse=True)

    keyword_pattern = (
        r"\b(?:"
        + "|".join(map(__import__("re").escape, keywords))
        + r")\b"
    )

    operator_pattern = (
        r"(?:"
        + "|".join(map(__import__("re").escape, operators))
        + r")"
    )

    return f"(?:{keyword_pattern}|{operator_pattern})"


def main():
    print(f"Loading tokenizer from:\n  {INPUT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(INPUT_DIR)

    backend = tokenizer.backend_tokenizer

    # Preserve Nemotron's original pre-tokenizer exactly.
    original_pre_tokenizer = backend.pre_tokenizer

    print("\nOriginal pre-tokenizer:")
    print(original_pre_tokenizer)

    python_tokens = get_python_tokens()
    python_pattern = build_python_pattern(python_tokens)

    print("\nPython boundary pattern:")
    print(python_pattern)

    python_split = pre_tokenizers.Split(
        Regex(python_pattern),
        behavior="isolated",
        invert=False,
    )

    # Python lexical boundaries first,
    # then Nemotron's original pre-tokenizer.
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            python_split,
            original_pre_tokenizer,
        ]
    )

    print("\nNew pre-tokenizer:")
    print(backend.pre_tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nSaved to:\n  {OUTPUT_DIR}")

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    tests = [
        "foo(x):",
        "def foo(x):",
        "x += 1",
        "x[y]",
        "x.y",
        "a = {x: 1}",
        "x **= 2",
        "x >>= 1",
        "if x >= 10:",
    ]

    print("\nPre-tokenization validation:")

    for text in tests:
        pieces = (
            backend.pre_tokenizer
            .pre_tokenize_str(text)
        )

        print(f"\n{text!r}")

        for piece, offsets in pieces:
            print(f"  {piece!r} {offsets}")

    print("\nFinal tokenization:")

    for text in tests:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
        )

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        print(f"\n{text!r}")
        print(tokens)


if __name__ == "__main__":
    main()