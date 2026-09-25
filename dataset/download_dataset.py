# sample_fineweb.py

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

SEED = 42
DATASET = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"
SPLIT = "train"

# FineWeb sample-10BT ~= 10B GPT-2 tokens.
#
# We want:
#
#   ~1.00B train
#   ~0.25B eval
#
# Total:
#
#   ~1.25B
#
# Therefore:
#
#   1.25 / 10 = 0.125
#
# We sample approximately 12.5% of the documents.
SAMPLE_PROBABILITY = 0.125

# Selected documents are split:
#
#   80% train
#   20% eval
#
TRAIN_RATIO = 0.80
OUTPUT_DIR = Path("data/fineweb_1.25B")
# Number of COMPLETE documents per Parquet shard.
DOCS_PER_SHARD = 50_000


def random_value(
    document_id: str,
    seed: int,
) -> float:
    """
    Deterministically map:
        document_id + seed to a random-looking number in [0, 1).

    This means the exact same FineWeb document will always
    receive the same random value.
    """
    data = f"{seed}:{document_id}".encode("utf-8")
    digest = hashlib.blake2b(
        data,
        digest_size=8,
    ).digest()
    value = int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )
    return value / 2**64


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        docs_per_shard: int,
    ):
        self.output_dir = output_dir / split
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.split = split
        self.docs_per_shard = docs_per_shard
        self.buffer = []
        self.shard_index = 0
        self.documents = 0
        self.tokens = 0

    def add(self, row):
        """
        Add ONE COMPLETE FineWeb document.
        """
        self.buffer.append(
            {
                "id": row["id"],
                "text": row["text"],
                "url": row["url"],
                "dump": row["dump"],
                "date": row["date"],
                "token_count": int(row["token_count"]),
            }
        )
        self.documents += 1
        self.tokens += int(row["token_count"])

        if len(self.buffer) >= self.docs_per_shard:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        table = pa.Table.from_pydict(
            {
                "id": [row["id"] for row in self.buffer],
                "text": [row["text"] for row in self.buffer],
                "url": [row["url"] for row in self.buffer],
                "dump": [row["dump"] for row in self.buffer],
                "date": [row["date"] for row in self.buffer],
                "token_count": [row["token_count"] for row in self.buffer],
            }
        )
        path = self.output_dir / (f"shard_" f"{self.shard_index:05d}.parquet")
        pq.write_table(
            table,
            path,
            compression="zstd",
        )
        shard_tokens = sum(row["token_count"] for row in self.buffer)

        print(
            f"[{self.split}] "
            f"{path} | "
            f"docs={len(self.buffer):,} | "
            f"tokens={shard_tokens:,}"
        )

        self.buffer.clear()
        self.shard_index += 1

    def close(self):
        self.flush()


def main():
    print("=" * 70)
    print("FineWeb → 1.25B-token approximate corpus")
    print("=" * 70)
    print()

    print(f"Dataset:       {DATASET}")
    print(f"Configuration: {CONFIG}")
    print(f"Seed:          {SEED}")
    print(f"Sample rate:   " f"{SAMPLE_PROBABILITY:.3f}")
    print(f"Train ratio:   " f"{TRAIN_RATIO:.2f}")
    print()

    dataset = load_dataset(
        DATASET,
        name=CONFIG,
        split=SPLIT,
        streaming=True,
    )
    train_writer = ShardWriter(
        OUTPUT_DIR,
        "train",
        DOCS_PER_SHARD,
    )
    eval_writer = ShardWriter(
        OUTPUT_DIR,
        "eval",
        DOCS_PER_SHARD,
    )

    source_documents = 0
    source_tokens = 0
    selected_documents = 0
    selected_tokens = 0

    for row in tqdm(dataset, desc="Streaming FineWeb"):
        source_documents += 1
        tokens = int(row["token_count"])
        source_tokens += tokens
        document_id = row["id"]

        # RANDOM SAMPLING
        #
        # Every document gets an independent deterministic
        # random value.
        #
        # The selection is distributed throughout the entire
        # FineWeb sample.

        selection_score = random_value(document_id, SEED)
        if selection_score >= SAMPLE_PROBABILITY:
            continue
        selected_documents += 1
        selected_tokens += tokens

        # TRAIN / EVAL ASSIGNMENT
        #
        # Independent deterministic random value.
        split_score = random_value(document_id, SEED + 1)
        if split_score < TRAIN_RATIO:
            train_writer.add(row)
        else:
            eval_writer.add(row)

    # Flush final partial shards.
    train_writer.close()
    eval_writer.close()

    # Final statistics.
    total_output_tokens = train_writer.tokens + eval_writer.tokens

    print()
    print("=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)
    print()

    print(f"Source documents:       " f"{source_documents:,}")
    print(f"Source GPT-2 tokens:    " f"{source_tokens:,}")
    print()

    print(f"Selected documents:     " f"{selected_documents:,}")
    print(f"Selected GPT-2 tokens:  " f"{selected_tokens:,}")
    print()

    print(f"Train documents:        " f"{train_writer.documents:,}")
    print(f"Train GPT-2 tokens:     " f"{train_writer.tokens:,}")
    print()

    print(f"Eval documents:         " f"{eval_writer.documents:,}")
    print(f"Eval GPT-2 tokens:      " f"{eval_writer.tokens:,}")
    print()

    print(f"TOTAL GPT-2 tokens:     " f"{total_output_tokens:,}")
    print()

    print(f"Output directory:       " f"{OUTPUT_DIR}")

    # Metadata.
    metadata = {
        "source": {
            "dataset": DATASET,
            "configuration": CONFIG,
            "split": SPLIT,
        },
        "sampling": {
            "method": ("deterministic " "document-level " "random sampling"),
            "seed": SEED,
            "document_selection_probability": (SAMPLE_PROBABILITY),
            "train_probability": TRAIN_RATIO,
            "eval_probability": (1.0 - TRAIN_RATIO),
            "preserve_complete_documents": True,
            "text_trimming": False,
            "text_splitting": False,
        },
        "expected": {
            "total_gpt2_tokens": 1_250_000_000,
            "train_gpt2_tokens": 1_000_000_000,
            "eval_gpt2_tokens": 250_000_000,
        },
        "actual": {
            "source_documents": source_documents,
            "source_gpt2_tokens": source_tokens,
            "selected_documents": selected_documents,
            "selected_gpt2_tokens": selected_tokens,
            "train_documents": (train_writer.documents),
            "train_gpt2_tokens": (train_writer.tokens),
            "eval_documents": (eval_writer.documents),
            "eval_gpt2_tokens": (eval_writer.tokens),
            "total_gpt2_tokens": (total_output_tokens),
        },
        "token_count": {
            "definition": ("FineWeb token_count " "using GPT-2 tokenizer"),
        },
        "format": {
            "type": "Parquet",
            "compression": "zstd",
        },
        "schema": {
            "id": "string",
            "text": "string",
            "url": "string",
            "dump": "string",
            "date": "string",
            "token_count": "int32",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("metadata.json written.")


if __name__ == "__main__":
    main()
