import argparse
import json
import logging
import time
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq
import requests
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi, HfFileSystem, get_token
from joblib import Parallel, delayed
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOG = logging.getLogger(__name__)


def load_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return tokenizer


def _num_rows_from_builder(dataset, config, split, token):
    try:
        builder = load_dataset_builder(dataset, name=config, token=token)
        if builder.info.splits and split in builder.info.splits:
            return builder.info.splits[split].num_examples
    except Exception:
        pass
    return None


def _num_rows_from_server(dataset, config, split, token):
    try:
        response = requests.get(
            "https://datasets-server.huggingface.co/size",
            params={"dataset": dataset},
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        partial = data.get("partial", False)
        total = 0
        for item in data["size"]["splits"]:
            if item["split"] != split:
                continue
            if config is not None and item["config"] != config:
                continue
            if partial and "estimated_num_rows" in item:
                total += item["estimated_num_rows"]
            else:
                total += item["num_rows"]
        return total or None
    except Exception:
        return None


def _num_rows_from_parquet(dataset, config, token):
    api = HfApi(token=token)
    parquet_paths: list[str] = []

    for item in tqdm(
        api.list_repo_tree(dataset, recursive=True, repo_type="dataset"),
        desc="Listing files",
        unit="file",
    ):
        if not hasattr(item, "path") or not item.path.endswith(".parquet"):
            continue
        if config is None or item.path.startswith(f"{config}/"):
            parquet_paths.append(item.path)

    if not parquet_paths:
        raise ValueError(f"No parquet files found for {dataset} config={config}")

    filesystem = HfFileSystem(token=token)

    def read_rows(path: str) -> int:
        return pq.read_metadata(
            f"datasets/{dataset}/{path}",
            filesystem=filesystem,
        ).num_rows

    counts = cast(
        list[int],
        list(
            Parallel(n_jobs=16, backend="threading")(
                delayed(read_rows)(path)
                for path in tqdm(
                    parquet_paths,
                    desc="Reading metadata",
                    unit="file",
                )
            )
        ),
    )
    return sum(counts)


def get_num_rows(dataset, config, split, token):
    LOG.info("Fetching row count for %s (config=%s, split=%s)", dataset, config, split)

    rows = _num_rows_from_builder(dataset, config, split, token)
    if rows is not None:
        LOG.info("Row count from dataset metadata: %s", f"{rows:,}")
        return rows

    rows = _num_rows_from_server(dataset, config, split, token)
    if rows is not None:
        LOG.info("Row count from datasets server: %s", f"{rows:,}")
        return rows

    rows = _num_rows_from_parquet(dataset, config, token)
    LOG.info("Row count from parquet metadata: %s", f"{rows:,}")
    return rows


def estimate_average_tokens(
    tokenizer,
    dataset,
    config,
    split,
    text_key,
    token,
    batch_size,
    threshold,
    min_docs,
    max_docs,
):
    kwargs = {
        "streaming": True,
        "split": split,
        "token": token,
    }
    if config:
        kwargs["name"] = config

    LOG.info("Streaming %s (config=%s, split=%s)...", dataset, config, split)
    stream = load_dataset(dataset, **kwargs)

    total_tokens = 0
    total_docs = 0
    previous_average = None
    batch = []
    started = time.time()
    progress = tqdm(total=max_docs, desc=dataset, unit="doc")

    eos_tokens = int(tokenizer.eos_token is not None)

    def process_batch(texts):
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_count = sum(
            len(input_ids) + eos_tokens
            for input_ids in encoded["input_ids"]
        )
        return token_count, len(texts)

    for row in stream:
        batch.append(row[text_key])
        if len(batch) < batch_size:
            continue

        batch_tokens, batch_docs = process_batch(batch)
        total_tokens += batch_tokens
        total_docs += batch_docs
        batch = []

        average = total_tokens / total_docs
        progress.update(batch_docs)
        progress.set_postfix(
            avg_tokens=f"{average:.0f}",
            documents=f"{total_docs:,}",
            docs_per_second=f"{total_docs / (time.time() - started):.0f}",
        )

        if total_docs >= min_docs:
            if previous_average is not None:
                relative_change = abs(average - previous_average) / average
                if relative_change < threshold:
                    progress.close()
                    return average, total_docs
            previous_average = average

        if total_docs >= max_docs:
            break

    if batch:
        batch_tokens, batch_docs = process_batch(batch)
        total_tokens += batch_tokens
        total_docs += batch_docs

    progress.close()
    if total_docs == 0:
        raise ValueError(f"No documents were read from {dataset}")
    return total_tokens / total_docs, total_docs


def human_number(value):
    if value >= 1e12:
        return f"{value / 1e12:.1f}T"
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{value / 1e3:.1f}K"


def main():
    parser = argparse.ArgumentParser(
        description="Estimate tokens for every source in a DataTrove config."
    )
    parser.add_argument("--sources", type=Path, default=Path("config/source.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/source_with_estimates.json"),
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended_II",
    )
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--min-docs", type=int, default=10_000)
    parser.add_argument("--max-docs", type=int, default=100_000)
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("Authenticate with Hugging Face before estimating tokens.")

    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    results = json.loads(json.dumps(sources))
    tokenizer = load_tokenizer(args.tokenizer)

    for index, (source_id, source) in enumerate(sources.items(), start=1):
        options = source["dataset_options"]
        dataset = str(source["dataset"])
        config = options.get("name")
        split = options["split"]
        text_key = source["text_key"]

        LOG.info("[%s/%s] Estimating %s", index, len(sources), source_id)
        num_rows = get_num_rows(dataset, config, split, token)
        average, sampled = estimate_average_tokens(
            tokenizer,
            dataset,
            config,
            split,
            text_key,
            token,
            args.batch_size,
            args.threshold,
            args.min_docs,
            args.max_docs,
        )
        estimated_tokens = round(average * num_rows)

        results[source_id].update(
            {
                "avg_tokens_per_doc": average,
                "sampled_docs": sampled,
                "num_rows": num_rows,
                "estimated_tokens": estimated_tokens,
            }
        )

        LOG.info(
            "%s: %s rows, %.0f tokens/doc, %s estimated tokens",
            source_id,
            f"{num_rows:,}",
            average,
            human_number(estimated_tokens),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    LOG.info("Saved estimates to %s", args.output)


if __name__ == "__main__":
    main()
