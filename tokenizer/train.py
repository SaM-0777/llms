import argparse
import json
import keyword
import tokenize as py_tokenize
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from qwen38.tokenization import Qwen38Tokenizer

DATASET_NAME = "nvidia/OpenCodeInstruct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("tokenizer"))
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--experiment-samples", type=int, default=20)
    return parser.parse_args()


def calculate_tokenizer_metrics(tokenizer, dataset, num_samples):
    total_tokens = 0
    total_characters = 0
    total_bytes = 0
    total_samples = 0
    max_tokens = 0

    start = time.perf_counter()

    progress = tqdm(
        total=num_samples,
        desc="Evaluating tokenizer",
        unit="sample",
        dynamic_ncols=True,
    )

    for index, sample in enumerate(dataset):
        if index >= num_samples:
            break

        text = f"{sample['input']}\n{sample['output']}"
        encoding = tokenizer.encode(text)

        token_count = len(encoding.ids)
        character_count = len(text)
        byte_count = len(text.encode("utf-8"))

        total_samples += 1
        total_tokens += token_count
        total_characters += character_count
        total_bytes += byte_count
        max_tokens = max(max_tokens, token_count)

        elapsed = time.perf_counter() - start
        samples_per_sec = total_samples / elapsed if elapsed else 0.0
        tokens_per_sec = total_tokens / elapsed if elapsed else 0.0

        progress.update(1)
        progress.set_postfix(
            {
                "samples/s": f"{samples_per_sec:.2f}",
                "tokens/s": f"{tokens_per_sec:.2f}",
                "tokens": f"{total_tokens:,}",
            }
        )

    progress.close()

    elapsed = time.perf_counter() - start

    return {
        "samples": total_samples,
        "total_tokens": total_tokens,
        "total_characters": total_characters,
        "total_bytes": total_bytes,
        "avg_tokens_per_sample": (
            total_tokens / total_samples if total_samples else 0.0
        ),
        "max_tokens_per_sample": max_tokens,
        "tokens_per_character": (
            total_tokens / total_characters if total_characters else 0.0
        ),
        "tokens_per_byte": (total_tokens / total_bytes if total_bytes else 0.0),
        "bytes_per_token": (total_bytes / total_tokens if total_tokens else 0.0),
        "compression_ratio": (total_bytes / total_tokens if total_tokens else 0.0),
        "samples_per_sec": (total_samples / elapsed if elapsed else 0.0),
        "tokens_per_sec": (total_tokens / elapsed if elapsed else 0.0),
        "elapsed_seconds": elapsed,
    }


def validate_pretokenized_tokens(tokenizer):
    keywords = list(keyword.kwlist) + list(keyword.softkwlist)
    operators = list(py_tokenize.EXACT_TOKEN_TYPES.keys())

    lexical_tokens = keywords + operators

    results = []
    single_token = 0
    split_tokens = 0
    missing_tokens = 0

    for lexical_token in lexical_tokens:
        encoding = tokenizer.encode(lexical_token)

        token_ids = encoding.ids
        tokens = encoding.tokens

        if len(token_ids) == 1:
            status = "single_token"
            single_token += 1
        elif len(token_ids) == 0:
            status = "missing"
            missing_tokens += 1
        else:
            status = "split"
            split_tokens += 1

        results.append(
            {
                "text": lexical_token,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "tokens": tokens,
                "status": status,
            }
        )

    return {
        "total": len(lexical_tokens),
        "single_token": single_token,
        "split": split_tokens,
        "missing": missing_tokens,
        "tokens": results,
    }


def text_iterator(dataset, num_samples, batch_size, stats, progress):
    batch = []
    start = time.perf_counter()

    for index, sample in enumerate(dataset):
        if index >= num_samples:
            break

        batch.append(f"{sample['input']}\n{sample['output']}")
        stats["samples"] += 1

        if len(batch) == batch_size:
            elapsed = time.perf_counter() - start
            samples_per_sec = stats["samples"] / elapsed if elapsed else 0.0
            progress.update(len(batch))
            progress.set_postfix({"samples/s": f"{samples_per_sec:.2f}"})

            yield batch
            batch = []

    if batch:
        elapsed = time.perf_counter() - start
        samples_per_sec = stats["samples"] / elapsed if elapsed else 0.0

        progress.update(len(batch))
        progress.set_postfix({"samples/s": f"{samples_per_sec:.2f}"})

        yield batch


def print_metrics(metrics):
    print("\nTokenizer evaluation")
    print("-" * 50)
    print(f"Samples:             {metrics['samples']:,}")
    print(f"Total tokens:        {metrics['total_tokens']:,}")
    print(f"Total characters:    {metrics['total_characters']:,}")
    print(f"Total bytes:         {metrics['total_bytes']:,}")
    print(f"Avg tokens/sample:   {metrics['avg_tokens_per_sample']:.2f}")
    print(f"Max tokens/sample:   {metrics['max_tokens_per_sample']:,}")
    print(f"Tokens/character:    {metrics['tokens_per_character']:.6f}")
    print(f"Tokens/byte:         {metrics['tokens_per_byte']:.6f}")
    print(f"Bytes/token:         {metrics['bytes_per_token']:.4f}")
    print(f"Compression ratio:   {metrics['compression_ratio']:.4f}")
    print(f"Samples/sec:         {metrics['samples_per_sec']:.2f}")
    print(f"Tokens/sec:          {metrics['tokens_per_sec']:.2f}")
    print(f"Elapsed:             {metrics['elapsed_seconds']:.2f}s")


def print_pretokenized_validation(validation):
    print("\nPython lexical token validation")
    print("-" * 50)
    print(f"Total:               {validation['total']:,}")
    print(f"Single token:        {validation['single_token']:,}")
    print(f"Split:               {validation['split']:,}")
    print(f"Missing:             {validation['missing']:,}")

    if validation["split"]:
        print("\nSplit lexical tokens:")
        for item in validation["tokens"]:
            if item["status"] == "split":
                print(f"  {item['text']!r} -> " f"{item['tokens']}")

    if validation["missing"]:
        print("\nMissing lexical tokens:")
        for item in validation["tokens"]:
            if item["status"] == "missing":
                print(f"  {item['text']!r}")


def main():
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than 0")

    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be greater than 0")

    if args.min_frequency <= 0:
        raise ValueError("--min-frequency must be greater than 0")

    if args.experiment_samples <= 0:
        raise ValueError("--experiment-samples must be greater than 0")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    stats = {"samples": 0}

    progress = tqdm(
        total=args.num_samples,
        desc="Training tokenizer",
        unit="sample",
        dynamic_ncols=True,
    )

    start = time.perf_counter()

    tokenizer = Qwen38Tokenizer.train(
        text_iterator(
            dataset,
            args.num_samples,
            args.batch_size,
            stats,
            progress,
        ),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<|endoftext|>"],
    )

    progress.close()

    elapsed = time.perf_counter() - start
    samples_per_sec = stats["samples"] / elapsed if elapsed else 0.0

    vocab_size = len(tokenizer._tokenizer.get_vocab())

    print("\nTokenizer training summary")
    print("-" * 50)
    print(f"Samples:             {stats['samples']:,}")
    print(f"Vocabulary size:     {vocab_size:,}")
    print(f"Requested vocab:     {args.vocab_size:,}")
    print(f"Min frequency:       {args.min_frequency}")
    print(f"Samples/sec:         {samples_per_sec:.2f}")
    print(f"Elapsed:             {elapsed:.2f}s")

    metrics = None
    validation = None
    experiment_samples = []

    if args.experiment:
        metrics_dataset = load_dataset(
            DATASET_NAME,
            split="train",
            streaming=True,
        )

        metrics = calculate_tokenizer_metrics(
            tokenizer,
            metrics_dataset,
            args.experiment_samples,
        )

        print_metrics(metrics)

        validation = validate_pretokenized_tokens(tokenizer)

        print_pretokenized_validation(validation)

        experiment_dataset = load_dataset(
            DATASET_NAME,
            split="train",
            streaming=True,
        )

        for index, sample in enumerate(experiment_dataset):
            if index >= args.experiment_samples:
                break

            text = f"{sample['input']}\n{sample['output']}"
            encoding = tokenizer.encode(text)

            experiment_samples.append(
                {
                    "id": sample.get("id"),
                    "input": sample["input"],
                    "output": sample["output"],
                    "token_count": len(encoding.ids),
                    "token_ids": encoding.ids,
                    "tokens": encoding.tokens,
                    "decoded": tokenizer._tokenizer.decode(encoding.ids),
                }
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)

        experiment_file = args.output_dir / "tokenizer_experiment.json"

        with experiment_file.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "dataset": DATASET_NAME,
                    "training_samples": stats["samples"],
                    "vocab_size": vocab_size,
                    "requested_vocab_size": args.vocab_size,
                    "min_frequency": args.min_frequency,
                    "metrics": metrics,
                    "pretokenized_tokens": validation,
                    "samples": experiment_samples,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\nExperiment:          {experiment_file}")

    if args.save:
        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        tokenizer.save(str(args.output_dir))

        print(f"Tokenizer saved:     " f"{args.output_dir}")


if __name__ == "__main__":
#  python train_tokenizer.py \
#  --num-samples 5000000 \
#  --vocab-size 16000 \
#  --min-frequency 2 \
#  --batch-size 1000 \
#  --save \
#  --output-dir tokenizer_5M
    main()
