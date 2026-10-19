from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import psutil


DEFAULT_TOKENIZER = "tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended_II"
ENGINES = ("megatron", "gigatoken")
TEXT_TEMPLATES = (
    "def add(a: int, b: int) -> int:\n    return a + b\n",
    "The derivative of x squared is 2x, and the integral of 2x is x squared.\n",
    "Implement a stable softmax function for a batch of logits.\n",
    "class Attention:\n    def forward(self, query, key, value):\n        return query @ key.T @ value\n",
    "Let A be a matrix in R^(m x n). The singular values describe its rank and conditioning.\n",
)


def build_text(index: int) -> str:
    return f"{TEXT_TEMPLATES[index % len(TEXT_TEMPLATES)]}# sample {index}\n"


def build_batch(start: int, size: int, documents: int) -> list[str]:
    end = min(start + size, documents)
    return [build_text(index) for index in range(start, end)]


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == "darwin" else int(value) * 1024


def resolve_tokenizer(path: str) -> tuple[Path, Path]:
    tokenizer_path = Path(path).expanduser()
    if tokenizer_path.is_dir():
        return tokenizer_path, tokenizer_path / "tokenizer.json"
    return tokenizer_path.parent, tokenizer_path


def load_eos_token(tokenizer_dir: Path) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.eos_token is None:
        raise ValueError(f"No EOS token configured in {tokenizer_dir}")
    return tokenizer.eos_token


def load_megatron_tokenizer(
    tokenizer_json: Path,
    eos_token: str | None,
    batch_size: int,
):
    from datatrove.pipeline.tokens.megatron_tokenizer import (
        MegatronDocumentTokenizer,
    )

    step = MegatronDocumentTokenizer(
        output_folder=".",
        tokenizer_name_or_path=str(tokenizer_json),
        eos_token=eos_token,
        batch_size=batch_size,
    )
    return step.tokenizer


def load_gigatoken_tokenizer(tokenizer_json: Path):
    import gigatoken as gt

    return gt.Tokenizer(str(tokenizer_json))


def benchmark(
    engine: str,
    tokenizer_json: Path,
    documents: int,
    batch_size: int,
    repeats: int,
    warmup_batches: int,
    eos_token: str | None,
    threads: int,
) -> dict[str, object]:
    if engine == "megatron":
        tokenizer = load_megatron_tokenizer(tokenizer_json, eos_token, batch_size)

        def encode(batch: list[str]) -> int:
            encoded = tokenizer.encode_batch(batch)
            return sum(len(item.ids) for item in encoded)

    else:
        tokenizer = load_gigatoken_tokenizer(tokenizer_json)

        def encode(batch: list[str]) -> int:
            encoded = tokenizer.encode_batch_list(batch, parallel=True)
            extra_tokens = 1 if eos_token is not None else 0
            return sum(len(item) + extra_tokens for item in encoded)

    batch_count = (documents + batch_size - 1) // batch_size

    for batch_number in range(min(warmup_batches, batch_count)):
        encode(build_batch(batch_number * batch_size, batch_size, documents))

    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_before = peak_rss_bytes()
    cpu_before = process.cpu_times()
    input_bytes = sum(
        len(build_text(index).encode("utf-8")) for index in range(documents)
    )

    started = time.perf_counter()
    total_tokens = 0
    for _ in range(repeats):
        for batch_number in range(batch_count):
            total_tokens += encode(
                build_batch(batch_number * batch_size, batch_size, documents)
            )
    elapsed = time.perf_counter() - started

    cpu_after = process.cpu_times()
    rss_after = process.memory_info().rss
    peak_rss = max(peak_before, peak_rss_bytes())
    cpu_seconds = (
        cpu_after.user
        + cpu_after.system
        - cpu_before.user
        - cpu_before.system
    )
    logical_cpus = psutil.cpu_count(logical=True) or 1
    total_input_bytes = input_bytes * repeats

    return {
        "engine": engine,
        "status": "ok",
        "tokenizer": str(tokenizer_json),
        "documents": documents,
        "repeats": repeats,
        "batch_size": batch_size,
        "threads": threads,
        "eos_enabled": eos_token is not None,
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "input_mb_per_second": total_input_bytes / elapsed / 1_000_000,
        "documents_per_second": documents * repeats / elapsed,
        "cpu_seconds": cpu_seconds,
        "cpu_percent_of_machine": 100 * cpu_seconds / elapsed / logical_cpus,
        "cpu_cores_used": cpu_seconds / elapsed,
        "rss_start_mb": rss_before / 1_000_000,
        "rss_end_mb": rss_after / 1_000_000,
        "peak_rss_mb": peak_rss / 1_000_000,
        "rss_delta_mb": (rss_after - rss_before) / 1_000_000,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the DataTrove MegatronDocumentTokenizer path."
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--documents", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Tokenizer threads; 0 uses the logical CPU count.",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Measure tokenization without the EOS used by dataset/mixture.py.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=ENGINES,
        default=list(ENGINES),
        help="Run one engine or both for comparison.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-engine", choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer-json", help=argparse.SUPPRESS)
    parser.add_argument("--eos-token", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_child(
    args: argparse.Namespace,
    engine: str,
    threads: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--child-engine",
        engine,
        "--tokenizer-json",
        args.tokenizer_json,
        "--documents",
        str(args.documents),
        "--batch-size",
        str(args.batch_size),
        "--repeats",
        str(args.repeats),
        "--warmup-batches",
        str(args.warmup_batches),
        "--threads",
        str(threads),
    ]
    if args.eos_token is not None:
        command.extend(["--eos-token", args.eos_token])

    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(threads)
    environment["RAYON_RS_NUM_THREADS"] = str(threads)
    environment["TOKENIZERS_PARALLELISM"] = "true"

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "engine": engine,
            "status": "error",
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }

    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    return {
        "engine": engine,
        "status": "error",
        "error": "Benchmark child returned no result.",
    }


def main() -> int:
    args = parse_args()

    if args.child:
        result = benchmark(
            engine=args.child_engine,
            tokenizer_json=Path(args.tokenizer_json),
            documents=args.documents,
            batch_size=args.batch_size,
            repeats=args.repeats,
            warmup_batches=args.warmup_batches,
            eos_token=args.eos_token,
            threads=args.threads,
        )
        print(json.dumps(result))
        return 0

    if args.documents < 1 or args.batch_size < 1 or args.repeats < 1:
        raise ValueError("documents, batch-size, and repeats must be positive")

    tokenizer_dir, tokenizer_json = resolve_tokenizer(args.tokenizer)
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)

    eos_token = None if args.no_eos else load_eos_token(tokenizer_dir)
    args.tokenizer_json = str(tokenizer_json)
    args.eos_token = eos_token
    threads = args.threads or (os.cpu_count() or 1)

    print(f"Tokenizer: {tokenizer_dir}")
    print(
        f"Documents: {args.documents:,} | batch size: {args.batch_size:,} | "
        f"repeats: {args.repeats} | threads: {threads} | "
        f"EOS: {'enabled' if eos_token else 'disabled'}"
    )
    print(
        "Megatron path: DataTrove MegatronDocumentTokenizer -> "
        "tokenizers.Tokenizer.encode_batch"
    )

    results = [run_child(args, engine, threads) for engine in args.engines]

    print("\nResults:")
    for result in results:
        if result["status"] != "ok":
            print(f"{result['engine']}: ERROR: {result['error']}")
            continue
        print(
            f"{result['engine']}: "
            f"{result['tokens_per_second']:,.0f} tokens/s | "
            f"{result['input_mb_per_second']:,.1f} MB/s | "
            f"CPU {result['cpu_percent_of_machine']:.1f}% "
            f"({result['cpu_cores_used']:.2f} cores) | "
            f"peak RSS {result['peak_rss_mb']:.1f} MB | "
            f"{result['elapsed_seconds']:.2f}s"
        )

    successful = [result for result in results if result["status"] == "ok"]
    if len(successful) == 2:
        megatron = next(result for result in successful if result["engine"] == "megatron")
        gigatoken = next(result for result in successful if result["engine"] == "gigatoken")
        print(
            f"Speed ratio: Gigatoken is "
            f"{gigatoken['tokens_per_second'] / megatron['tokens_per_second']:.2f}x "
            "the Megatron/Hugging Face path."
        )

    if args.json_output:
        print(json.dumps(results, indent=2))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
