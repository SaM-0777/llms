import argparse
import os
from datetime import datetime, timezone

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronDocumentTokenizer

REPO_ID = "SamLegion888/SWE_test_1"
SOURCES = {
    "philosophy_data_project": (
        "yjkim27/The-Philosophy-Data-Project",
        {"split": "train"},
        None,
        None,
        "sentence_str",
        1.0,
    ),
    "philosophy_corpus": (
        "LisaMegaWatts/philosophy-corpus",
        {"split": "train"},
        None,
        None,
        "text",
        0.10,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer",
        required=True,
        default="tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended",
        help="Path to your local tokenizer.json file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=512,
        help="Maximum rows read per source; use 0 for the full source.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Documents tokenized per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--run-id",
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "philosophy-test-%Y%m%d-%H%M%S"
    )

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
    )

    for name, (
        dataset_id,
        dataset_options,
        _estimated_tokens,
        _schema,
        text_key,
        sample_rate,
    ) in SOURCES.items():
        print(f"Source: {dataset_id}")
        print(f"Text field: {text_key}")
        print(f"Sampling rate: {sample_rate:.0%}")
        print(f"Row limit: {args.limit or 'unlimited'}")

        output_folder = f"hf://datasets/{REPO_ID}/" f"runs/{run_id}/{name}/"
        logging_dir = f"hf://datasets/{REPO_ID}/" f"_datatrove_logs/{run_id}/{name}/"

        pipeline = [
            HuggingFaceDatasetReader(
                dataset=dataset_id,
                dataset_options=dataset_options,
                streaming=True,
                limit=args.limit if args.limit > 0 else -1,
                text_key=text_key,
                batch_size=args.batch_size,
                doc_progress=True,
            ),
            SamplerFilter(
                rate=sample_rate,
                seed=args.seed,
            ),
            MegatronDocumentTokenizer(
                output_folder=output_folder,
                save_filename=name,
                tokenizer_name_or_path=args.tokenizer,
                eos_token=args.eos_token,
                batch_size=args.batch_size,
            ),
        ]

        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=logging_dir,
            tasks=1,
            workers=1,
        )

        executor.run()

        print(f"Finished source: {name}")
        print(f"Output: {output_folder}")

    print("\nExperiment finished.")
    print(f"Repository: https://huggingface.co/datasets/{REPO_ID}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
