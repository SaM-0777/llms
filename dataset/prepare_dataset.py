import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOKEN_DTYPE = np.uint32
TOKEN_DTYPE_NAME = "uint32"
TOKEN_BYTES = np.dtype(TOKEN_DTYPE).itemsize


@dataclass
class SplitStats:
    name: str
    examples: int = 0
    tokens: int = 0
    skipped_examples: int = 0
    elapsed_seconds: float = 0.0

    @property
    def examples_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0

        return self.examples / self.elapsed_seconds

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0

        return self.tokens / self.elapsed_seconds

    @property
    def average_tokens_per_example(self) -> float:
        if self.examples == 0:
            return 0.0

        return self.tokens / self.examples


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    config: str | None
    revision: str | None

    train_split: str
    test_split: str | None

    test_ratio: float
    split_seed: int

    tokenizer_name: str

    text_column: str | None
    text_columns: tuple[str, ...] | None
    text_template: str | None
    separator: str

    batch_size: int
    max_examples: int | None
    max_length: int | None
    append_eos: bool

    output_dir: Path
    overwrite: bool


# Argument parsing
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a Hugging Face dataset, tokenize it using a fast "
            "Hugging Face tokenizer, and generate train.bin/test.bin."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset", required=True, help="Hugging Face dataset ID."
    )
    dataset_group.add_argument(
        "--config", default=None, help="Dataset configuration/subset name."
    )
    dataset_group.add_argument(
        "--revision",
        default=None,
        help="Dataset revision, branch, tag, or commit hash.",
    )

    # Splits
    split_group = parser.add_argument_group("splits")
    split_group.add_argument(
        "--train-split", default="train", help="Source split used for training."
    )
    split_group.add_argument(
        "--test-split",
        default=None,
        help=(
            "Existing test/validation split. " "If omitted, --test-ratio can be used."
        ),
    )
    split_group.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of the source train split assigned to test. "
            "Used only when --test-split is not provided."
        ),
    )
    split_group.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Deterministic seed for train/test assignment.",
    )

    # Tokenizer
    tokenizer_group = parser.add_argument_group("tokenizer")
    tokenizer_group.add_argument(
        "--tokenizer",
        required=True,
        help=(
            "Tokenizer path or Hugging Face tokenizer ID. "
            "Use your Py-lexical Nemotron tokenizer here."
        ),
    )
    tokenizer_group.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Optional maximum number of tokens per example. "
            "No truncation when omitted."
        ),
    )
    tokenizer_group.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append EOS to every encoded example.",
    )

    # Text extraction
    text_group = parser.add_argument_group("text extraction")
    extraction = text_group.add_mutually_exclusive_group(required=True)
    extraction.add_argument(
        "--text-column",
        default=None,
        help="Single dataset field containing the text.",
    )
    extraction.add_argument(
        "--text-columns",
        default=None,
        help=("Comma-separated fields to concatenate. " "Example: prompt,response"),
    )
    extraction.add_argument(
        "--text-template",
        default=None,
        help=(
            "Python format template using dataset fields. "
            'Example: "{prompt}\\n\\n{response}"'
        ),
    )
    text_group.add_argument(
        "--separator",
        default="\n",
        help="Separator used by --text-columns.",
    )

    # Processing
    processing_group = parser.add_argument_group("processing")
    processing_group.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of examples tokenized per batch.",
    )
    processing_group.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples processed from each source stream.",
    )

    # Output
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing train.bin, test.bin and dataset.json.",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing output directory.",
    )

    return parser


# Argument validation
def build_config(args: argparse.Namespace) -> DatasetConfig:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be greater than zero.")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("--max-length must be greater than zero.")
    if not 0.0 <= args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be in the range [0, 1).")
    if args.test_split is not None and args.test_ratio > 0:
        raise ValueError("Specify either --test-split or --test-ratio, not both.")

    text_columns = None
    if args.text_columns is not None:
        columns = tuple(
            column.strip() for column in args.text_columns.split(",") if column.strip()
        )
        if not columns:
            raise ValueError("--text-columns must contain at least one column.")

        text_columns = columns

    return DatasetConfig(
        dataset=args.dataset,
        config=args.config,
        revision=args.revision,
        train_split=args.train_split,
        test_split=args.test_split,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        tokenizer_name=args.tokenizer,
        text_column=args.text_column,
        text_columns=text_columns,
        text_template=args.text_template,
        separator=args.separator,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        max_length=args.max_length,
        append_eos=args.append_eos,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


def load_streaming_dataset(config: DatasetConfig, split: str):
    kwargs: dict[str, Any] = {"path": config.dataset, "split": split, "streaming": True}

    if config.config is not None:
        kwargs["name"] = config.config
    if config.revision is not None:
        kwargs["revision"] = config.revision

    return load_dataset(**kwargs)


def get_field(example: dict[str, Any], field: str) -> Any:
    value: Any = example

    for part in field.split("."):
        if not isinstance(value, dict):
            raise TypeError(
                f"Cannot resolve '{field}': "
                f"'{part}' is being accessed on "
                f"{type(value).__name__}."
            )

        if part not in value:
            raise KeyError(f"Dataset field '{field}' does not exist.")

        value = value[part]

    return value


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def extract_text(
    example: dict[str, Any],
    config: DatasetConfig,
) -> str:
    # Single column
    if config.text_column is not None:
        return stringify(get_field(example, config.text_column))

    # Multiple columns
    if config.text_columns is not None:
        values = [
            stringify(get_field(example, column)) for column in config.text_columns
        ]
        return config.separator.join(values)

    # Template
    if config.text_template is not None:

        class ExampleValues(dict):
            def __missing__(self, key: str) -> str:
                raise KeyError(
                    f"Dataset field '{key}' referenced by "
                    "--text-template does not exist."
                )

        values = ExampleValues(
            {key: stringify(value) for key, value in example.items()}
        )

        return config.text_template.format_map(values)

    raise RuntimeError("No text extraction configuration was provided.")


# Deterministic train/test assignment
def is_test_example(
    index: int,
    *,
    ratio: float,
    seed: int,
) -> bool:
    if ratio <= 0:
        return False

    payload = f"{seed}:{index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    random_value = int.from_bytes(digest, byteorder="big", signed=False)
    threshold = int(ratio * (2**64))
    return random_value < threshold


# Binary writer
class BinaryTokenWriter:
    """
    Atomic append-only writer for raw uint32 token IDs.

    Data is first written to:
        train.bin.tmp

    and renamed to:
        train.bin

    only after the complete write succeeds.
    """

    def __init__(self, path: Path):
        self.path = path
        self.temp_path = Path(f"{path}.tmp")
        self.file = self.temp_path.open("wb", buffering=1024 * 1024)
        self.tokens_written = 0

    def write(self, token_ids: Iterable[int]) -> int:
        array = np.asarray(list(token_ids), dtype=TOKEN_DTYPE)
        if array.size == 0:
            return 0

        array.tofile(self.file)
        count = int(array.size)
        self.tokens_written += count

        return count

    def write_batch(self, token_batches: Iterable[Iterable[int]]) -> int:
        total = 0
        for token_ids in token_batches:
            total += self.write(token_ids)

        return total

    def commit(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        os.replace(self.temp_path, self.path)

    def abort(self) -> None:
        try:
            self.file.close()
        finally:
            self.temp_path.unlink(missing_ok=True)


def tokenize_batch(
    tokenizer, texts: list[str], config: DatasetConfig
) -> list[list[int]]:
    kwargs: dict[str, Any] = {
        "add_special_tokens": False,
    }

    if config.max_length is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = config.max_length

    encoded = tokenizer(texts, **kwargs)
    token_ids = encoded["input_ids"]

    if config.append_eos:
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError(
                "--append-eos was specified but the tokenizer "
                "does not define an EOS token."
            )

        token_ids = [ids + [eos_id] for ids in token_ids]

    return token_ids


def process_split(
    dataset,
    *,
    tokenizer,
    writer: BinaryTokenWriter,
    config: DatasetConfig,
    split_name: str,
    example_filter: Callable[[int, dict[str, Any]], bool] | None = None,
) -> SplitStats:
    stats = SplitStats(name=split_name)
    start = time.perf_counter()
    texts: list[str] = []
    progress = tqdm(desc=split_name, unit=" examples", dynamic_ncols=True)

    def flush() -> None:
        if not texts:
            return

        token_batches = tokenize_batch(tokenizer, texts, config)

        for ids in token_batches:
            writer.write(ids)
            stats.tokens += len(ids)

        texts.clear()

    try:
        for source_index, example in enumerate(dataset):
            if (
                config.max_examples is not None
                and stats.examples + stats.skipped_examples >= config.max_examples
            ):
                break

            if example_filter is not None:
                if not example_filter(source_index, example):
                    continue

            text = extract_text(example, config)
            if not text:
                stats.skipped_examples += 1
                continue

            texts.append(text)
            stats.examples += 1

            if len(texts) >= config.batch_size:
                flush()

            elapsed = time.perf_counter() - start
            progress.update(1)

            if elapsed > 0:
                progress.set_postfix(
                    tokens=f"{stats.tokens:,}",
                    ex_s=f"{stats.examples / elapsed:,.1f}",
                    tok_s=f"{stats.tokens / elapsed:,.0f}",
                )

        flush()
    finally:
        progress.close()

    stats.elapsed_seconds = time.perf_counter() - start
    return stats


def prepare_output_directory(
    path: Path,
    overwrite: bool,
) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(
                f"Output path exists but is not a directory: {path}"
            )

        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}\n\n"
                "Use --overwrite if you want to reuse it."
            )

    path.mkdir(parents=True, exist_ok=True)


def tokenizer_metadata(tokenizer) -> dict[str, Any]:
    return {
        "name_or_path": tokenizer.name_or_path,
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
    }


def write_metadata(
    *,
    output_dir: Path,
    config: DatasetConfig,
    tokenizer,
    train_stats: SplitStats,
    test_stats: SplitStats,
) -> None:
    metadata = {
        "format": {
            "name": "raw_token_ids",
            "dtype": TOKEN_DTYPE_NAME,
            "bytes_per_token": TOKEN_BYTES,
            "endianness": "native",
        },
        "dataset": {
            "id": config.dataset,
            "config": config.config,
            "revision": config.revision,
        },
        "tokenizer": tokenizer_metadata(tokenizer),
        "configuration": {
            "train_split": config.train_split,
            "test_split": config.test_split,
            "test_ratio": config.test_ratio,
            "split_seed": config.split_seed,
            "text_column": config.text_column,
            "text_columns": (
                list(config.text_columns) if config.text_columns is not None else None
            ),
            "text_template": config.text_template,
            "separator": config.separator,
            "batch_size": config.batch_size,
            "max_examples": config.max_examples,
            "max_length": config.max_length,
            "append_eos": config.append_eos,
        },
        "train": asdict(train_stats),
        "test": asdict(test_stats),
        "files": {
            "train.bin": {
                "tokens": train_stats.tokens,
                "bytes": train_stats.tokens * TOKEN_BYTES,
            },
            "test.bin": {
                "tokens": test_stats.tokens,
                "bytes": test_stats.tokens * TOKEN_BYTES,
            },
        },
    }

    output_path = output_dir / "dataset.json"
    temp_path = Path(f"{output_path}.tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    os.replace(
        temp_path,
        output_path,
    )


def validate_tokenizer(tokenizer) -> None:
    vocab_size = tokenizer.vocab_size
    dtype_max = np.iinfo(TOKEN_DTYPE).max

    if vocab_size > dtype_max:
        raise ValueError(
            f"Tokenizer vocabulary size ({vocab_size:,}) exceeds "
            f"{TOKEN_DTYPE_NAME} capacity ({dtype_max:,})."
        )

    if tokenizer.is_fast is not True:
        raise ValueError("A fast tokenizer is required for this pipeline.")


def print_summary(
    *,
    config: DatasetConfig,
    tokenizer,
    train_stats: SplitStats,
    test_stats: SplitStats,
) -> None:
    print()
    print("=" * 72)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 72)

    print()
    print("Dataset")
    print("-" * 72)
    print(f"  ID             : {config.dataset}")

    if config.config is not None:
        print(f"  Config         : {config.config}")
    if config.revision is not None:
        print(f"  Revision       : {config.revision}")

    print()
    print("Tokenizer")
    print("-" * 72)
    print(f"  Path/ID        : {config.tokenizer_name}")
    print(f"  Vocabulary     : {tokenizer.vocab_size:,}")
    print(f"  Dtype          : {TOKEN_DTYPE_NAME}")
    print(f"  EOS ID         : {tokenizer.eos_token_id}")

    print()
    print("Train")
    print("-" * 72)
    print(f"  Examples       : {train_stats.examples:,}")
    print(f"  Skipped        : {train_stats.skipped_examples:,}")
    print(f"  Tokens         : {train_stats.tokens:,}")
    print(f"  Avg tokens/ex  : " f"{train_stats.average_tokens_per_example:,.2f}")
    print(f"  Examples/sec   : " f"{train_stats.examples_per_second:,.2f}")
    print(f"  Tokens/sec     : " f"{train_stats.tokens_per_second:,.0f}")
    print(
        f"  File size      : "
        f"{train_stats.tokens * TOKEN_BYTES / (1024**3):,.3f} GiB"
    )

    print()
    print("Test")
    print("-" * 72)
    print(f"  Examples       : {test_stats.examples:,}")
    print(f"  Skipped        : {test_stats.skipped_examples:,}")
    print(f"  Tokens         : {test_stats.tokens:,}")
    print(f"  Avg tokens/ex  : " f"{test_stats.average_tokens_per_example:,.2f}")
    print(f"  Examples/sec   : " f"{test_stats.examples_per_second:,.2f}")
    print(f"  Tokens/sec     : " f"{test_stats.tokens_per_second:,.0f}")
    print(
        f"  File size      : " f"{test_stats.tokens * TOKEN_BYTES / (1024**3):,.3f} GiB"
    )

    print()
    print("Output")
    print("-" * 72)
    print(f"  Directory      : {config.output_dir}")
    print(f"  Train          : {config.output_dir / 'train.bin'}")
    print(f"  Test           : {config.output_dir / 'test.bin'}")
    print(f"  Metadata       : {config.output_dir / 'dataset.json'}")
    print()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = build_config(args)
        prepare_output_directory(config.output_dir, config.overwrite)

        print("=" * 72)
        print("STREAMING DATASET PREPARATION")
        print("=" * 72)

        print()
        print("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
        validate_tokenizer(tokenizer)

        print(f"Tokenizer vocabulary: " f"{tokenizer.vocab_size:,}")

        # TRAIN
        print()
        print(f"Loading train split: " f"{config.train_split!r}")

        train_dataset = load_streaming_dataset(config, config.train_split)
        train_writer = BinaryTokenWriter(config.output_dir / "train.bin")

        try:
            train_filter: Callable[[int, dict[str, Any]], bool] | None = None
            if config.test_split is None and config.test_ratio > 0:

                def should_include_in_train(
                    index: int, _example: dict[str, Any]
                ) -> bool:
                    return not is_test_example(
                        index, ratio=config.test_ratio, seed=config.split_seed
                    )

                train_filter = should_include_in_train

            train_stats = process_split(
                train_dataset,
                tokenizer=tokenizer,
                writer=train_writer,
                config=config,
                split_name="train",
                example_filter=train_filter,
            )

            train_writer.commit()

        except Exception:
            train_writer.abort()
            raise

        if config.test_split is not None:
            print()
            print(f"Loading test split: " f"{config.test_split!r}")

            test_dataset = load_streaming_dataset(config, config.test_split)
            test_writer = BinaryTokenWriter(config.output_dir / "test.bin")

            try:
                test_stats = process_split(
                    test_dataset,
                    tokenizer=tokenizer,
                    writer=test_writer,
                    config=config,
                    split_name="test",
                )
                test_writer.commit()
            except Exception:
                test_writer.abort()
                raise

        elif config.test_ratio > 0:
            print()
            print("Re-streaming train split for deterministic " "test extraction...")

            test_dataset = load_streaming_dataset(config, config.train_split)
            test_writer = BinaryTokenWriter(config.output_dir / "test.bin")

            try:
                def test_filter(index: int, _example: dict[str, Any]) -> bool:
                    return is_test_example(
                        index, ratio=config.test_ratio, seed=config.split_seed
                    )

                test_stats = process_split(
                    test_dataset,
                    tokenizer=tokenizer,
                    writer=test_writer,
                    config=config,
                    split_name="test",
                    example_filter=test_filter,
                )

                test_writer.commit()

            except Exception:
                test_writer.abort()
                raise

        else:
            test_writer = BinaryTokenWriter(config.output_dir / "test.bin")
            try:
                test_writer.commit()
            except Exception:
                test_writer.abort()
                raise
            test_stats = SplitStats(
                name="test",
            )

        write_metadata(
            output_dir=config.output_dir,
            config=config,
            tokenizer=tokenizer,
            train_stats=train_stats,
            test_stats=test_stats,
        )
        print_summary(
            config=config,
            tokenizer=tokenizer,
            train_stats=train_stats,
            test_stats=test_stats,
        )
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
