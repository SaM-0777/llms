import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from qwen38.tokenization import Qwen38Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained tokenizer on OpenCodeInstruct."
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the saved tokenizer directory.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=50_000,
        help="Number of evaluation samples.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )

    return parser.parse_args()


def load_tokenizer(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {path}")

    tokenizer_json = path / "tokenizer.json"

    if not tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found in: {path}")

    tokenizer = Qwen38Tokenizer()
    tokenizer._tokenizer = tokenizer._tokenizer.from_file(str(tokenizer_json))

    return tokenizer


def evaluate(tokenizer, dataset, num_samples):
    sequence_lengths = []

    total_tokens = 0
    total_bytes = 0
    total_chars = 0
    samples = 0

    start = time.perf_counter()

    progress = tqdm(
        total=num_samples,
        desc="Evaluating tokenizer",
        unit="sample",
    )

    for sample in dataset:
        if samples >= num_samples:
            break

        text = f"{sample['input']}\n{sample['output']}"

        encoded = tokenizer.encode(text)
        token_count = len(encoded.ids)

        sequence_lengths.append(token_count)

        total_tokens += token_count
        total_bytes += len(text.encode("utf-8"))
        total_chars += len(text)

        samples += 1

        elapsed = time.perf_counter() - start

        samples_per_sec = samples / elapsed if elapsed > 0 else 0.0

        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

        progress.update(1)
        progress.set_postfix(
            {
                "samples/s": f"{samples_per_sec:.2f}",
                "tokens/s": f"{tokens_per_sec:.0f}",
            }
        )

    progress.close()

    if samples == 0:
        raise RuntimeError("No samples were evaluated.")

    elapsed = time.perf_counter() - start

    sequence_lengths.sort()

    def percentile(values, percentile):
        index = (len(values) - 1) * percentile
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)

        weight = index - lower

        return values[lower] + (values[upper] - values[lower]) * weight

    metrics = {
        "samples": samples,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "avg_tokens_per_sample": total_tokens / samples,
        "tokens_per_byte": (total_tokens / total_bytes if total_bytes else 0.0),
        "bytes_per_token": (total_bytes / total_tokens if total_tokens else 0.0),
        "chars_per_token": (total_chars / total_tokens if total_tokens else 0.0),
        "avg_sequence_length": total_tokens / samples,
        "p50_sequence_length": percentile(sequence_lengths, 0.50),
        "p90_sequence_length": percentile(sequence_lengths, 0.90),
        "p95_sequence_length": percentile(sequence_lengths, 0.95),
        "p99_sequence_length": percentile(sequence_lengths, 0.99),
        "max_sequence_length": max(sequence_lengths),
        "samples_per_sec": samples / elapsed if elapsed else 0.0,
        "tokens_per_sec": total_tokens / elapsed if elapsed else 0.0,
        "elapsed_seconds": elapsed,
    }

    return metrics


def print_metrics(tokenizer_path, metrics):
    print()
    print("=" * 60)
    print("Tokenizer Evaluation")
    print("=" * 60)

    print(f"Tokenizer:             {tokenizer_path}")
    print(f"Samples:               {metrics['samples']:,}")
    print(f"Total tokens:          {metrics['total_tokens']:,}")
    print(f"Total bytes:           {metrics['total_bytes']:,}")
    print(f"Total characters:      {metrics['total_chars']:,}")

    print()
    print("Tokenization efficiency")
    print("-" * 60)
    print(f"Tokens / sample:       " f"{metrics['avg_tokens_per_sample']:.4f}")
    print(f"Tokens / byte:         " f"{metrics['tokens_per_byte']:.6f}")
    print(f"Bytes / token:         " f"{metrics['bytes_per_token']:.6f}")
    print(f"Characters / token:    " f"{metrics['chars_per_token']:.6f}")

    print()
    print("Sequence length")
    print("-" * 60)
    print(f"Average:               " f"{metrics['avg_sequence_length']:.2f}")
    print(f"P50:                   " f"{metrics['p50_sequence_length']:.2f}")
    print(f"P90:                   " f"{metrics['p90_sequence_length']:.2f}")
    print(f"P95:                   " f"{metrics['p95_sequence_length']:.2f}")
    print(f"P99:                   " f"{metrics['p99_sequence_length']:.2f}")
    print(f"Max:                   " f"{metrics['max_sequence_length']:,}")

    print()
    print("Performance")
    print("-" * 60)
    print(f"Samples / sec:         " f"{metrics['samples_per_sec']:.2f}")
    print(f"Tokens / sec:          " f"{metrics['tokens_per_sec']:.2f}")
    print(f"Elapsed:               " f"{metrics['elapsed_seconds']:.2f}s")

    print("=" * 60)


def main():
    args = parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than 0")

    tokenizer = load_tokenizer(args.tokenizer)

    dataset = load_dataset(
        "nvidia/OpenCodeInstruct",
        split="train",
        streaming=True,
    )

    metrics = evaluate(
        tokenizer=tokenizer,
        dataset=dataset,
        num_samples=args.num_samples,
    )

    print_metrics(
        tokenizer_path=args.tokenizer,
        metrics=metrics,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metrics,
                f,
                indent=2,
            )

        print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()
