import argparse
from contextlib import contextmanager
from functools import partial
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
from collections.abc import Generator, Mapping
from typing import Any

from transformers import AutoTokenizer
from huggingface_hub import HfApi

from datatrove.io import DataFolderLike
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.megatron_tokenizer import (
    MegatronDocumentTokenizer,
    MegatronTokenizedFile,
)
from datatrove.utils.batching import batched
from datatrove.pipeline.readers.base import BaseReader

LOG = logging.getLogger("stream_preprocess")
RUN_ID = "swe-1-200b"
REPO_ID = "SamLegion888/SWE_test_1"
TOTAL_TOKEN_BUDGET = 200000000000 * 1.005
CHECKPOINT_REPO_PATH = "metadata/checkpoint.json"
CHECKPOINT_VERSION = 3
MIN_LOCAL_FREE_BYTES = 8 * 1024**3

SOURCES = {
    "nemotron_cc_code_v1": {
        "dataset": "nvidia/Nemotron-CC-Code-v1",
        "dataset_options": {"name": "data", "split": "train"},
        "text_key": "text",
        "ratio": 0.45,  # 45%
    },
    "code_v2_synthetic_question_answering": {
        "dataset": "nvidia/Nemotron-Pretraining-Code-v2",
        "dataset_options": {
            "name": "Synthetic-Question-Answering",
            "split": "train",
        },
        "text_key": "content",
        "ratio": 0.045,  # 4.5%
    },
    "code_v2_synthetic_code_review": {
        "dataset": "nvidia/Nemotron-Pretraining-Code-v2",
        "dataset_options": {
            "name": "Synthetic-Code-Review",
            "split": "train",
        },
        "text_key": "content",
        "ratio": 0.045,  # 4.5%
    },
    "code_v2_synthetic_student_teacher": {
        "dataset": "nvidia/Nemotron-Pretraining-Code-v2",
        "dataset_options": {
            "name": "Synthetic-Student-Teacher",
            "split": "train",
        },
        "text_key": "content",
        "ratio": 0.03,  # 3%
    },
    "code_v2_synthetic_rewriting": {
        "dataset": "nvidia/Nemotron-Pretraining-Code-v2",
        "dataset_options": {
            "name": "Synthetic-Rewriting",
            "split": "train",
        },
        "text_key": "content",
        "ratio": 0.015,  # 1.5%
    },
    "code_v2_synthetic_transpilation": {
        "dataset": "nvidia/Nemotron-Pretraining-Code-v2",
        "dataset_options": {
            "name": "Synthetic-Transpilation",
            "split": "train",
        },
        "text_key": "content",
        "ratio": 0.015,  # 1.5%
    },
    "ultra_fineweb_general_en": {
        "dataset": "openbmb/Ultra-FineWeb",
        "dataset_options": {"name": "default", "split": "en"},
        "text_key": "content",
        "ratio": 0.15,
    },
    "nemotron_math_textbooks": {
        "dataset": "nvidia/Nemotron-Pretraining-Specialized-v1",
        "dataset_options": {
            "name": "Nemotron-Pretraining-Math-Textbooks",
            "split": "train",
        },
        "text_key": "text",
        "ratio": 0.10,
    },
    "nemotron_scientific_coding": {
        "dataset": "nvidia/Nemotron-Pretraining-Specialized-v1",
        "dataset_options": {
            "name": "Nemotron-Pretraining-Scientific-Coding",
            "split": "train",
        },
        "text_key": "text",
        "ratio": 0.07,
    },
    "nemotron_infini_byte_reasoning": {
        "dataset": "nvidia/Nemotron-Pretraining-Specialized-v1",
        "dataset_options": {
            "name": "Nemotron-Pretraining-InfiniByte-Reasoning",
            "split": "train",
        },
        "text_key": "text",
        "ratio": 0.033,
    },
    "nemotron_rqa": {
        "dataset": "nvidia/Nemotron-Pretraining-Specialized-v1",
        "dataset_options": {
            "name": "Nemotron-Pretraining-RQA",
            "split": "train",
        },
        "text_key": "text",
        "ratio": 0.0167,
    },
    "yjkim27_the_philosophy_data_project": {
        "dataset": "yjkim27/The-Philosophy-Data-Project",
        "dataset_options": {"name": "default", "split": "train"},
        "text_key": "sentence_str",
        "ratio": 0.02,
    },
    "LisaMegaWatts_philosophy_corpus": {
        "dataset": "LisaMegaWatts/philosophy-corpus",
        "dataset_options": {"name": "default", "split": "train"},
        "text_key": "text",
        "ratio": 0.01,
    },
}


class LocalStorageSafetyStop(RuntimeError):
    """Raised before local storage reaches the reserved safety margin."""


class ShardWriter(MegatronDocumentTokenizer):
    def __init__(
        self,
        *args,
        output_folder: DataFolderLike,
        save_filename: str,
        tokenizer_name_or_path: str,
        eos_token: str,
        batch_size: int = 10000,
        upload_block_size: int | None = None,
        target_tokens: int,
        max_tokens_per_shard: int,
        cache_path: Path,
        **kwargs,
    ):
        super().__init__(
            output_folder,
            save_filename,
            tokenizer_name_or_path,
            eos_token,
            batch_size,
            upload_block_size,
        )

        self.target_tokens = target_tokens
        self.max_tokens_per_shard = max_tokens_per_shard
        self.cache_path = cache_path
        self.total_tokens = 0
        self.accepted_documents = 0
        self.shard_tokens: list[int] = []

    def run(self, data, rank: int = 0, world_size: int = 1):
        current: MegatronTokenizedFile | None = None
        current_tokens = 0
        shard_index = 1

        # Early stop if not enough device storage
        def check_storage() -> None:
            try:
                ensure_local_storage(self.cache_path)
            except LocalStorageSafetyStop:
                if current is not None:
                    current.close()
                    self.shard_tokens.append(current_tokens)
                raise

        def open_shard() -> MegatronTokenizedFile:
            check_storage()
            filename = f"{self.save_filename}-{shard_index:04d}"
            return MegatronTokenizedFile(
                self.output_folder,
                filename,
                upload_block_size=self.upload_block_size,
                token_size=self.token_size,
            )

        for batch in batched(data, self.batch_size):
            check_storage()

            if self.total_tokens >= self.target_tokens:
                break

            encoded_batch = self.tokenizer.encode_batch(
                [document.text for document in batch]
            )

            for encoded in encoded_batch:
                tokens = encoded.ids

                if not tokens:
                    continue

                if current is not None and current_tokens > self.max_tokens_per_shard:
                    current.close()
                    self.shard_tokens.append(current_tokens)
                    current = None
                    current_tokens = 0
                    shard_index += 1

                if current is None:
                    current = open_shard()

                current.write(tokens)
                current_tokens += len(tokens)
                self.total_tokens += len(tokens)
                self.accepted_documents += 1
                self.stat_update("tokens", value=len(tokens))

        if current is not None:
            current.close()
            self.shard_tokens.append(current_tokens)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer",
        default="tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended",
        help="Local Hugging Face tokenizer directory.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=4,
        help="Num tasks",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=(os.cpu_count() or 8) // 2,
        help="Num workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Documents per tokenization batch.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2048,  # 2GB
        help="Shard size in MiB.",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_budget_plan():
    ratio_budgets = {
        source_id: int(int(TOTAL_TOKEN_BUDGET) * str(source["ratio"]))
        for source_id, source in SOURCES.items()
    }

    return ratio_budgets


def calculate_tokens_per_shard(shard_size: int, dtype: str = "uint32"):
    if dtype == "uint16":
        num_bytes = 2
    elif dtype == "uint32":
        num_bytes = 4
    else:
        raise ValueError(f"Invalid dtype")

    avg_tokens_per_shard = shard_size * 1_000_000 / num_bytes
    return avg_tokens_per_shard


def load_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return tokenizer


def load_remote_checkpoint() -> dict[str, Any] | None:
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    path = f"datasets/{REPO_ID}/{CHECKPOINT_REPO_PATH}"
    try:
        with filesystem.open(path, "rb") as handle:
            return json.loads(handle.read())
    except FileNotFoundError:
        return None


def ensure_local_storage(path: Path) -> None:
    free_bytes = shutil.disk_usage(path).free
    if free_bytes < MIN_LOCAL_FREE_BYTES:
        raise LocalStorageSafetyStop(
            f"Local storage safety stop: {free_bytes / 1024**3:.2f} GiB free; "
            f"{MIN_LOCAL_FREE_BYTES / 1024**3:.2f} GiB is reserved"
        )


@contextmanager
def isolated_hf_cache() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(prefix="hf-stream-cache-") as temp_dir:
        cache_dir = Path(temp_dir)
        names = ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HF_XET_CACHE")
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = str(cache_dir / name.lower())
        os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        try:
            yield cache_dir
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def upload_json(api: Any, path_in_repo: str, payload: Mapping[str, Any]) -> None:
    from io import BytesIO

    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    api.upload_file(
        path_or_fileobj=BytesIO(data),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {path_in_repo}",
    )


def data_adapter(data, path, id_in_file, *, source_id, source):
    text_key = source["text_key"]
    id_key = f"{source_id}/{path}/{id_in_file}"
    return {
        "text": text_key,
        "id": id_key,
        "metadata": {"source_id": source_id, "dataset": source["dataset"]},
    }


def dataset_reader(source_id: str, source: dict, cache_dir: Path):
    dataset = source["dataset"]
    dataset_options = dict(source["dataset_options"])
    dataset_options["cache_dir"] = str(cache_dir)
    adapter = partial(data_adapter, source_id, source)
    return HuggingFaceDatasetReader(
        dataset=dataset,
        dataset_options=dataset_options,
        streaming=True,
        adapter=adapter,
    )


def run(args):
    budget_plan = build_budget_plan()
    cpu_count = max(1, os.cpu_count() or 1)
    batch_size = args.batch_size
    shard_size = args.shard_size

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    eos_token = tokenizer.eos_token
    tokenizer_json = Path(args.tokenizer).expanduser() / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(
            f"DataTrove requires the local tokenizer.json: {tokenizer_json}"
        )
    tokenizer_json = tokenizer_json.resolve()

    dtype = "uint16"
    if vocab_size > 2**16:
        dtype = "uint32"

    tokens_per_shard = calculate_tokens_per_shard(shard_size, dtype=dtype)

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    num_tasks = args.num_tasks
    num_workers = max(args.num_tasks, args.num_workers)
    resources = {
        "cpu_count": cpu_count,
        "batch_size": batch_size,
        "shard_size": shard_size,
        "tokens_per_shard": tokens_per_shard,
        "tasks": num_tasks,
        "workers": num_workers,
    }
    checkpoint_metadata = {
        "repo_id": REPO_ID,
        "tokenizer": str(tokenizer_json),
        "eos_token": eos_token,
        "budget_plan": budget_plan,
    }
    state = load_remote_checkpoint()
    if state is not None:
        if state.get("metadata") != checkpoint_metadata:
            raise ValueError(f"Checkpoint requires a fresh run: {CHECKPOINT_REPO_PATH}")
        results = state["results"]
        results["resources"] = resources
        completed_sources = set(state.get("completed_sources", []))
        LOG.info(
            "resuming completed_sources=%d checkpoint=%s",
            len(completed_sources),
            CHECKPOINT_REPO_PATH,
        )
    else:
        results = {
            "run_id": RUN_ID,
            "repo_id": REPO_ID,
            "tokenizer": str(tokenizer_json),
            "eos_token": eos_token,
            "resources": resources,
            "budget_plan": budget_plan,
            "sources": {},
        }
        completed_sources = set()

    with isolated_hf_cache() as cache_dir:
        ensure_local_storage(cache_dir)

        for source_id, source in SOURCES.items():
            if source_id in completed_sources:
                continue

            target_tokens = int(budget_plan[source_id])
            writer = ShardWriter(
                output_folder=f"hf://datasets/{REPO_ID}",
                save_filename=source_id,
                tokenizer_name_or_path=str(tokenizer_json),
                eos_token=eos_token,
                batch_size=resources["batch_size"],
                target_tokens=target_tokens,
                max_tokens_per_shard=resources["shard_tokens"],
                cache_path=cache_dir,
            )
            pipeline = [
                dataset_reader(source_id=source_id, source=source, cache_dir=cache_dir),
                writer,
            ]

            log_dir = f"hf://datasets/{REPO_ID}/logs/{RUN_ID}/{source_id}"
            LOG.info(
                "source=%s dataset=%s target_tokens=%s output=root",
                source_id,
                source["dataset"],
                f"{target_tokens:,}",
            )

            LocalPipelineExecutor(
                pipeline=pipeline,
                logging_dir=str(log_dir),
                tasks=resources["tasks"],
                workers=resources["workers"],
            ).run()

            source_result = {
                "dataset": source["dataset"],
                "ratio": source["ratio"],
                "target_tokens": target_tokens,
                "realized_tokens": writer.total_tokens,
                "accepted_documents": writer.accepted_documents,
                "shard_tokens": writer.shard_tokens,
            }
            results["sources"][source_id] = source_result
            completed_sources.add(source_id)

            upload_json(
                api,
                CHECKPOINT_REPO_PATH,
                {
                    "version": CHECKPOINT_VERSION,
                    "metadata": checkpoint_metadata,
                    "completed_sources": sorted(completed_sources),
                    "results": results,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    results["realized_total_tokens"] = sum(
        int(source_result["realized_tokens"])
        for source_result in results["sources"].values()
    )
    upload_json(api, f"metadata/{RUN_ID}.json", results)
    upload_json(api, "metadata/mixture.json", {"sources": SOURCES, **budget_plan})
    LOG.info("realized_total_tokens=%s", f"{results['realized_total_tokens']:,}")
    return results


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    args = parse_args()

    try:
        run(args=args)
        return 0
    except LocalStorageSafetyStop as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
