import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from transformers import AutoTokenizer


@dataclass
class Args:
    bin_file: str = "data/Nemotron-SFT-SWE-v3.5/qwen35-extended/tokens.bin"
    metadata_file: str = "data/Nemotron-SFT-SWE-v3.5/qwen35-extended/metadata.json"
    tokenizer: str = "tokenizer/Qwen35-4B-vocab-extended"
    output: str = "data/Nemotron-SFT-SWE-v3.5/qwen35-extended/token_distribution.csv"
    chunk_tokens: int = 5_000_000
    top_k: int = 100


def main(args: Args):
    metadata = json.loads(Path(args.metadata_file).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = np.dtype(metadata["dtype"])
    counts = np.zeros(len(tokenizer), dtype=np.uint64)
    total_tokens = 0

    with Path(args.bin_file).open("rb") as token_file:
        while True:
            ids = np.fromfile(token_file, dtype=dtype, count=args.chunk_tokens)
            if len(ids) == 0:
                break
            counts += np.bincount(
                ids.astype(np.int64),
                minlength=len(tokenizer),
            ).astype(np.uint64)
            total_tokens += len(ids)

    observed_ids = np.flatnonzero(counts)
    observed_ids = observed_ids[np.argsort(counts[observed_ids])[::-1]]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["rank", "token_id", "token", "count", "frequency"])
        for rank, token_id in enumerate(observed_ids, start=1):
            count = int(counts[token_id])
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            writer.writerow(
                [
                    rank,
                    int(token_id),
                    token,
                    count,
                    count / total_tokens,
                ]
            )

    print(f"tokens: {total_tokens:,}")
    print(f"unique tokens: {len(observed_ids):,}")
    print(f"distribution: {output}")
    print()
    print("rank  token_id  count       frequency  token")
    for rank, token_id in enumerate(observed_ids[: args.top_k], start=1):
        count = int(counts[token_id])
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        print(f"{rank:4}  {int(token_id):8}  {count:10,}  {count / total_tokens:10.6%}  {token!r}")


if __name__ == "__main__":
    main(tyro.cli(Args))
