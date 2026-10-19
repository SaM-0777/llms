import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

try:
    from tokenizer.lexical_matching import (
        add_missing_lexical_tokens,
        install_lexical_matcher,
        model_fingerprint,
        patch_saved_added_token_metadata,
        serialized_pre_tokenizer,
    )
except ModuleNotFoundError:
    from lexical_matching import (
        add_missing_lexical_tokens,
        install_lexical_matcher,
        model_fingerprint,
        patch_saved_added_token_metadata,
        serialized_pre_tokenizer,
    )

#MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
MODEL_ID = "Qwen/Qwen3.5-4B"

DEFAULT_INPUT = Path("tokenizer/notation_lexicals_II.json")

DEFAULT_OUTPUT = Path("tokenizer/Qwen35-4B-vocab-extended")


def load_unified_tokens(path: Path) -> list[str]:
    """Load and validate the unified lexical inventory."""

    if not path.exists():
        raise FileNotFoundError(f"Lexical inventory not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    tokens = data.get("unified")

    if not isinstance(tokens, list):
        raise ValueError(f"'unified' must be an array in {path}")

    if not all(isinstance(token, str) for token in tokens):
        raise ValueError("Every entry in 'unified' must be a string.")

    # Remove duplicate and empty strings.
    return sorted({token for token in tokens if token})


def update_existing_tokenizer(output: Path, tokens: list[str]) -> None:
    """Apply matching to an existing extended tokenizer in place."""

    if not output.exists():
        raise FileNotFoundError(f"Existing tokenizer directory not found: {output}")

    report_path = output / "extension_report.json"
    previous_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    original_vocab = tokenizer.get_vocab()
    original_model_fingerprint = model_fingerprint(tokenizer)
    original_pre_tokenizer = serialized_pre_tokenizer(tokenizer)
    first_added_token_id = previous_report.get(
        "original_vocabulary_size",
        min(original_vocab.values()),
    )

    lexical_pattern = install_lexical_matcher(tokenizer, tokens)
    tokenizer.save_pretrained(output)
    metadata_entries_updated = patch_saved_added_token_metadata(
        output / "tokenizer.json",
        tokens,
        first_added_token_id,
    )

    updated = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    updated_vocab = updated.get_vocab()
    changed_ids = {
        token: (token_id, updated_vocab.get(token))
        for token, token_id in original_vocab.items()
        if updated_vocab.get(token) != token_id
    }
    if changed_ids:
        raise RuntimeError(
            "Existing tokenizer IDs changed during in-place update: "
            f"{list(changed_ids.items())[:10]}"
        )
    if model_fingerprint(updated) != original_model_fingerprint:
        raise RuntimeError("The existing BPE model changed during in-place update.")

    report = dict(previous_report)
    report["extended_vocabulary_size"] = len(updated)
    report["existing_token_ids_preserved"] = True
    report["matching_updated_in_place"] = True
    report["lexical_matching"] = {
        "strategy": "boundary-aware lexical split followed by the original pre-tokenizer",
        "longest_match_first": True,
        "lexical_regex_length": len(lexical_pattern),
        "base_bpe_model_sha256": original_model_fingerprint,
        "extended_bpe_model_sha256": model_fingerprint(updated),
        "original_pre_tokenizer": original_pre_tokenizer,
        "added_token_metadata_entries_updated": metadata_entries_updated,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Existing tokenizer updated in place.")
    print(f"Vocabulary size: {len(updated)}")
    print(f"Added-token metadata entries updated: {metadata_entries_updated}")
    print("Existing token IDs preserved: YES")
    print("BPE model preserved: YES")
    print(f"Tokenizer saved to: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extend the official Nemotron tokenizer with lexical "
            "strings missing from its vocabulary."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON file containing the unified token array.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where the extended tokenizer will be saved.",
    )

    parser.add_argument(
        "--base-model",
        default=MODEL_ID,
        help="Official Nemotron model ID or local tokenizer directory.",
    )

    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Apply boundary matching to the tokenizer already in --output.",
    )

    args = parser.parse_args()

    # 1. Load lexical strings.
    tokens = load_unified_tokens(args.input)

    print(f"Lexical strings loaded: {len(tokens)}")

    if args.update_existing:
        update_existing_tokenizer(args.output, tokens)
        return

    # 2. Load the official base tokenizer.
    print(f"\nLoading base tokenizer: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # Snapshot the original vocabulary and token IDs.
    original_vocab = tokenizer.get_vocab()
    original_vocab_size = len(tokenizer)
    original_model_fingerprint = model_fingerprint(tokenizer)
    original_pre_tokenizer = serialized_pre_tokenizer(tokenizer)

    print(f"Original vocabulary size: {original_vocab_size}")

    # 3. Find strings that are not already vocabulary entries.
    missing_tokens = [token for token in tokens if token not in original_vocab]

    existing_tokens = len(tokens) - len(missing_tokens)

    print(f"Strings already in vocabulary: {existing_tokens}")
    print(f"Missing strings to add: {len(missing_tokens)}")

    # 4. Add only missing strings with category-aware matching properties.
    # Hugging Face assigns new IDs without renumbering existing IDs.
    missing_tokens, added_count = add_missing_lexical_tokens(tokenizer, tokens)

    if added_count != len(missing_tokens):
        raise RuntimeError(
            "Unexpected number of added tokens: "
            f"expected {len(missing_tokens)}, got {added_count}"
        )

    # 5. Verify all original vocabulary IDs remain unchanged.
    extended_vocab = tokenizer.get_vocab()

    changed_ids = {
        token: (token_id, extended_vocab.get(token))
        for token, token_id in original_vocab.items()
        if extended_vocab.get(token) != token_id
    }

    if changed_ids:
        raise RuntimeError(
            "Existing vocabulary IDs changed. "
            f"Examples: {list(changed_ids.items())[:10]}"
        )

    # 6. Verify each added string is recognized as one token.
    failed_tokens = []

    for token in missing_tokens:
        token_ids = tokenizer.encode(
            token,
            add_special_tokens=False,
        )

        if len(token_ids) != 1:
            failed_tokens.append((token, token_ids))

    if failed_tokens:
        raise RuntimeError(
            f"{len(failed_tokens)} added strings did not encode "
            "as a single token. Examples: "
            f"{failed_tokens[:10]}"
        )

    # 7. Install lexical matching before the original pre-tokenizer.
    lexical_pattern = install_lexical_matcher(tokenizer, tokens)

    if model_fingerprint(tokenizer) != original_model_fingerprint:
        raise RuntimeError("The official BPE model changed during extension.")

    # 8. Save the extended tokenizer separately.
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output)

    # 9. Save a report for reproducibility.
    report = {
        "base_model": args.base_model,
        "input_json": str(args.input),
        "output_directory": str(args.output),
        "original_vocabulary_size": original_vocab_size,
        "extended_vocabulary_size": len(tokenizer),
        "lexical_strings_loaded": len(tokens),
        "strings_already_in_vocabulary": existing_tokens,
        "new_tokens_added": added_count,
        "existing_token_ids_preserved": True,
        "all_added_tokens_encode_as_single_token": True,
        "lexical_matching": {
            "strategy": "boundary-aware lexical split followed by the original pre-tokenizer",
            "longest_match_first": True,
            "lexical_regex_length": len(lexical_pattern),
            "base_bpe_model_sha256": original_model_fingerprint,
            "extended_bpe_model_sha256": model_fingerprint(tokenizer),
            "original_pre_tokenizer": original_pre_tokenizer,
        },
        "added_tokens": [
            {
                "token": token,
                "id": tokenizer.convert_tokens_to_ids(token),
            }
            for token in missing_tokens
        ],
    }

    report_path = args.output / "extension_report.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("VOCABULARY EXTENSION COMPLETE")
    print("=" * 60)
    print(f"Original vocabulary size: {original_vocab_size}")
    print(f"Extended vocabulary size: {len(tokenizer)}")
    print(f"New tokens added: {added_count}")
    print("Existing token IDs preserved: YES")
    print("Added tokens encode individually: YES")
    print(f"Tokenizer saved to: {args.output.resolve()}")
    print(f"Report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
