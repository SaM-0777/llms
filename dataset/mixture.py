import argparse
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from huggingface_hub import HfApi, get_token
from transformers import AutoTokenizer

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronDocumentTokenizer

LOG = logging.getLogger("datatrove_mixture")


def load_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream, sample, and tokenize dataset sources with DataTrove."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default="config/source.json",
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--total-tokens", type=int, default=200_000_000_000)
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended_II",
    )
    parser.add_argument("--num-tasks", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oversample", type=float, default=1.05)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.num_tasks < 1 or args.num_workers < 1:
        raise ValueError("--num-tasks and --num-workers must be positive")
    if args.num_workers > args.num_tasks:
        raise ValueError("--num-workers cannot exceed --num-tasks")
    if args.oversample <= 0:
        raise ValueError("--oversample must be positive")

    token = get_token()
    if not token:
        raise RuntimeError("Authenticate with Hugging Face before starting the run.")

    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false" if args.num_workers > 1 else "true",
    )

    sources = json.loads(args.sources.read_text(encoding="utf-8"))

    repo_id = args.repo_id
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)
    eos_token = tokenizer.eos_token
    tokenizer_json = Path(args.tokenizer) / "tokenizer.json"

    ratio_total = sum(Decimal(str(source["ratio"])) for source in sources.values())
    planned_tokens = int(Decimal(args.total_tokens) * ratio_total)
    shortfall = args.total_tokens - planned_tokens

    LOG.info(
        "repo=%s total_tokens=%s ratio_total=%s planned_tokens=%s shortfall=%s",
        args.repo_id,
        f"{args.total_tokens:,}",
        ratio_total,
        f"{planned_tokens:,}",
        f"{shortfall:,}",
    )

    for source_id, source in sources.items():
        text_key = source["text_key"]
        estimated_tokens = source["estimated_tokens"]
        estimated_tokens = int(estimated_tokens)

        target_tokens = int(Decimal(args.total_tokens) * Decimal(str(source["ratio"])))
        rate = min(
            1.0,
            float(
                Decimal(target_tokens)
                * Decimal(str(args.oversample))
                / Decimal(estimated_tokens)
            ),
        )

        dataset_options = dict(source["dataset_options"])
        dataset_options["token"] = token
        dataset_options["columns"] = [text_key]

        LOG.info(
            "source=%s target_tokens=%s estimated_tokens=%s sample_rate=%.8f",
            source_id,
            f"{target_tokens:,}",
            f"{estimated_tokens:,}",
            rate,
        )

        pipeline = [
            HuggingFaceDatasetReader(
                dataset=str(source["dataset"]),
                dataset_options=dataset_options,
                streaming=True,
                text_key=text_key,
                id_key="id",
            ),
            SamplerFilter(rate=rate, seed=args.seed),
            MegatronDocumentTokenizer(
                output_folder=f"hf://datasets/{repo_id}/{source_id}",
                save_filename=source_id,
                tokenizer_name_or_path=str(tokenizer_json),
                eos_token=eos_token,
                batch_size=args.batch_size,
            ),
        ]

        LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=args.num_tasks,
            workers=args.num_workers,
            logging_dir=(
                f"hf://datasets/{repo_id}/logs/{run_id}/{source_id}"
            ),
            skip_completed=True,
        ).run()


if __name__ == "__main__":
    main()
