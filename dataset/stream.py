import argparse
from contextlib import contextmanager
from decimal import Decimal
from functools import partial
import json
import logging
from multiprocess.managers import SyncManager
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

LOG = logging.getLogger("stream_preprocess")
CHECKPOINT_REPO_PATH = "metadata/checkpoint.json"
CHECKPOINT_VERSION = 3
MIN_LOCAL_FREE_BYTES = 8 * 1024**3


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
        budget_state: Mapping[str, Any],
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
        self.budget_state = budget_state
        self.total_tokens = 0
        self.accepted_documents = 0
        self.shards: list[dict[str, Any]] = []

    def run(self, data, rank: int = 0, world_size: int = 1):
        current: MegatronTokenizedFile | None = None
        current_tokens = 0

        def check_storage() -> None:
            nonlocal current, current_tokens
            try:
                ensure_local_storage(self.cache_path)
            except LocalStorageSafetyStop:
                if current is not None:
                    current.close()
                    self.shards.append(
                        {"filename": current.filename, "tokens": current_tokens}
                    )
                    current = None
                    current_tokens = 0
                raise

        def open_shard() -> MegatronTokenizedFile:
            check_storage()
            with self.budget_state["lock"]:
                shard_index = self.budget_state["next_shard"].value
                self.budget_state["next_shard"].value += 1
            filename = f"{self.save_filename}-{shard_index:04d}"
            return MegatronTokenizedFile(
                self.output_folder,
                filename,
                upload_block_size=self.upload_block_size,
                token_size=self.token_size,
            )

        try:
            stop = False
            for batch in batched(data, self.batch_size):
                check_storage()

                with self.budget_state["lock"]:
                    if self.budget_state["total_tokens"].value >= self.target_tokens:
                        break

                encoded_batch = self.tokenizer.encode_batch(
                    [document.text for document in batch]
                )

                with self.budget_state["lock"]:
                    remaining_tokens = (
                        self.target_tokens - self.budget_state["total_tokens"].value
                    )
                    accepted_batch = []
                    reserved_tokens = 0
                    if remaining_tokens > 0:
                        for encoded in encoded_batch:
                            if not encoded.ids:
                                continue
                            accepted_batch.append(encoded)
                            reserved_tokens += len(encoded.ids)
                            if reserved_tokens >= remaining_tokens:
                                break
                        self.budget_state["total_tokens"].value += reserved_tokens
                    stop = (
                        reserved_tokens >= remaining_tokens
                        if remaining_tokens > 0
                        else True
                    )

                for encoded in accepted_batch:
                    tokens = encoded.ids

                    if (
                        current is not None
                        and current_tokens >= self.max_tokens_per_shard
                    ):
                        current.close()
                        self.shards.append(
                            {"filename": current.filename, "tokens": current_tokens}
                        )
                        current = None
                        current_tokens = 0

                    if current is None:
                        current = open_shard()

                    current.write(tokens)
                    current_tokens += len(tokens)
                    self.total_tokens += len(tokens)
                    self.accepted_documents += 1
                    self.stat_update("tokens", value=len(tokens))

                if stop:
                    break
        finally:
            if current is not None:
                current.close()
                self.shards.append(
                    {"filename": current.filename, "tokens": current_tokens}
                )
            self.budget_state["stats"][rank] = {
                "tokens": self.total_tokens,
                "accepted_documents": self.accepted_documents,
                "shards": list(self.shards),
            }


def get_tokenizer_name(tokenizer_path: str) -> str:
    normalized = str(tokenizer_path).replace("\\", "/").rstrip("/")
    if normalized.endswith("/tokenizer.json"):
        normalized = normalized[: -len("/tokenizer.json")]
    parts = [part for part in normalized.split("/") if part]
    if "tokenizer" in parts:
        return "/".join(parts[parts.index("tokenizer") :])
    return parts[-1] if parts else "tokenizer"


class Executor(LocalPipelineExecutor):
    def __init__(self, *, executor_metadata: Mapping[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.executor_metadata = dict(executor_metadata)

    def save_executor_as_json(self, indent: int = 4):
        payload = {
            **self.executor_metadata,
            "executor": "LocalPipelineExecutor",
            "skip_completed": self.skip_completed,
            "tasks": self.tasks,
            "workers": self.workers,
            "start_method": self.start_method,
            "local_tasks": self.local_tasks,
            "local_rank_offset": self.local_rank_offset,
            "world_size": self.world_size,
        }
        with self.logging_dir.open("executor.json", "w") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sources",
        type=str,
        default="config/source.json",
        help="Source json file",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        required=True,
        help="Total token budget",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Huggingface repo_id",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended",
        help="Local Hugging Face tokenizer directory.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Total DataTrove tasks. Defaults to the available CPU count.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 2),
        help="Concurrent DataTrove workers. Defaults to CPU count minus two.",
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
    return parser.parse_args()


def get_sources(file: str | Path) -> dict[str, Any]:
    file_path = Path(file)

    if not file_path.exists():
        raise FileNotFoundError("Sources json file path does not exists")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_budget_plan(total_tokens: int, sources: dict):
    ratio_budgets = {
        source_id: int(int(total_tokens) * Decimal(str(source["ratio"])))
        for source_id, source in sources.items()
    }

    return ratio_budgets


def calculate_tokens_per_shard(shard_size: int, dtype: str = "uint32"):
    if dtype == "uint16":
        num_bytes = 2
    elif dtype == "uint32":
        num_bytes = 4
    else:
        raise ValueError(f"Invalid dtype")

    avg_tokens_per_shard = shard_size * 1024**2 // num_bytes
    return avg_tokens_per_shard


def load_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return tokenizer


def load_remote_checkpoint(repo_id: str) -> dict[str, Any] | None:
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    path = f"datasets/{repo_id}/{CHECKPOINT_REPO_PATH}"
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
def isolated_hf_cache(*, parallel_workers: int) -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(prefix="hf-stream-cache-") as temp_dir:
        cache_dir = Path(temp_dir)
        names = ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HF_XET_CACHE")
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = str(cache_dir / name.lower())
        os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false" if parallel_workers > 1 else "true"
        )
        try:
            yield cache_dir
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def upload_json(
    api: Any, repo_id: str, path_in_repo: str, payload: Mapping[str, Any]
) -> None:
    from io import BytesIO

    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    api.upload_file(
        path_or_fileobj=BytesIO(data),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {path_in_repo}",
    )


def data_adapter(reader, data, path, id_in_file, *, source_id, source):
    text_key = source["text_key"]
    id_key = f"{source_id}/{path}/{id_in_file}"
    return {
        "text": data.get(text_key, ""),
        "id": id_key,
        "metadata": {"source_id": source_id, "dataset": source["dataset"]},
    }


def dataset_reader(source_id: str, source: dict, cache_dir: Path):
    dataset = source["dataset"]
    dataset_options = dict(source["dataset_options"])
    dataset_options["cache_dir"] = str(cache_dir)
    adapter = partial(data_adapter, source_id=source_id, source=source)
    return HuggingFaceDatasetReader(
        dataset=dataset,
        dataset_options=dataset_options,
        streaming=True,
        adapter=adapter,
        text_key="text",
        id_key="id",
    )


def run(args):
    repo_id = args.repo_id
    run_id = f"{repo_id}-v1"
    sources = get_sources(args.sources)
    total_tokens = args.total_tokens * 1.005
    budget_plan = build_budget_plan(total_tokens=total_tokens, sources=sources)
    cpu_count = max(1, (os.cpu_count() or 2) // 2)
    batch_size = args.batch_size
    shard_size = args.shard_size

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    eos_token = tokenizer.eos_token
    tokenizer_json = Path(args.tokenizer) / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(
            f"DataTrove requires the local tokenizer.json: {tokenizer_json}"
        )
    tokenizer_json = tokenizer_json.resolve()
    tokenizer_name = get_tokenizer_name(args.tokenizer)

    dtype = "uint16"
    if vocab_size > 2**16:
        dtype = "uint32"

    tokens_per_shard = calculate_tokens_per_shard(shard_size, dtype=dtype)

    api = HfApi()
    try:
        identity = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication is required. "
            "Run hf auth login or set HF_TOKEN, then run this script again."
        ) from exc

    LOG.info("authenticated with Hugging Face as %s", identity.get("name", "unknown"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    num_tasks = max(1, args.num_tasks)
    num_workers = min(num_tasks, max(1, args.num_workers))
    resources = {
        "cpu_count": cpu_count,
        "batch_size": batch_size,
        "shard_size": shard_size,
        "tokens_per_shard": tokens_per_shard,
        "tasks": num_tasks,
        "workers": num_workers,
    }
    checkpoint_metadata = {
        "repo_id": repo_id,
        "tokenizer": tokenizer_name,
        "eos_token": eos_token,
        "budget_plan": budget_plan,
    }
    state = load_remote_checkpoint(repo_id=repo_id)
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
            "run_id": run_id,
            "repo_id": repo_id,
            "tokenizer": tokenizer_name,
            "eos_token": eos_token,
            "resources": resources,
            "budget_plan": budget_plan,
            "sources": {},
        }
        completed_sources = set()

    manager: Any = SyncManager()
    manager.start()
    try:
        with isolated_hf_cache(parallel_workers=resources["workers"]) as cache_dir:
            ensure_local_storage(cache_dir)

            for source_id, source in sources.items():
                if source_id in completed_sources:
                    continue

                target_tokens = int(budget_plan[source_id])
                budget_state = {
                    "lock": manager.Lock(),
                    "total_tokens": manager.Value("q", 0),
                    "next_shard": manager.Value("i", 1),
                    "stats": manager.dict(),
                }
                writer = ShardWriter(
                    output_folder=f"hf://datasets/{repo_id}",
                    save_filename=source_id,
                    tokenizer_name_or_path=str(tokenizer_json),
                    eos_token=eos_token,
                    batch_size=resources["batch_size"],
                    target_tokens=target_tokens,
                    max_tokens_per_shard=resources["tokens_per_shard"],
                    cache_path=cache_dir,
                    budget_state=budget_state,
                )
                pipeline = [
                    dataset_reader(
                        source_id=source_id, source=source, cache_dir=cache_dir
                    ),
                    writer,
                ]

                log_dir = f"hf://datasets/{repo_id}/logs/{source_id}"
                executor_metadata = {
                    "run_id": run_id,
                    "source_id": source_id,
                    "pipeline": [
                        {
                            "type": "HuggingFaceDatasetReader",
                            "dataset": source["dataset"],
                            "dataset_options": dict(source["dataset_options"]),
                            "cache_dir": "hf-stream-cache",
                            "streaming": True,
                            "batch_size": 1000,
                            "text_key": source["text_key"],
                            "id_key": "id",
                        },
                        {
                            "type": "ShardWriter",
                            "tokenizer": tokenizer_name,
                            "eos_token": eos_token,
                            "output_folder": f"hf://datasets/{repo_id}",
                            "save_filename": source_id,
                            "batch_size": resources["batch_size"],
                            "target_tokens": target_tokens,
                            "max_tokens_per_shard": resources["tokens_per_shard"],
                            "cache_path": "hf-stream-cache",
                        },
                    ],
                    "logging_dir": log_dir,
                }
                LOG.info(
                    "source=%s dataset=%s target_tokens=%s tasks=%d workers=%d output=root",
                    source_id,
                    source["dataset"],
                    f"{target_tokens:,}",
                    resources["tasks"],
                    resources["workers"],
                )

                Executor(
                    executor_metadata=executor_metadata,
                    pipeline=pipeline,
                    logging_dir=str(log_dir),
                    tasks=resources["tasks"],
                    workers=resources["workers"],
                    start_method="fork",
                    skip_completed=False,
                ).run()

                task_results = list(budget_state["stats"].values())
                shards = sorted(
                    (
                        shard
                        for task_result in task_results
                        for shard in task_result["shards"]
                    ),
                    key=lambda shard: shard["filename"],
                )
                source_result = {
                    "dataset": source["dataset"],
                    "ratio": source["ratio"],
                    "target_tokens": target_tokens,
                    "realized_tokens": int(budget_state["total_tokens"].value),
                    "accepted_documents": sum(
                        task_result["accepted_documents"]
                        for task_result in task_results
                    ),
                    "shards": shards,
                    "shard_tokens": [shard["tokens"] for shard in shards],
                }
                results["sources"][source_id] = source_result
                completed_sources.add(source_id)

                upload_json(
                    api,
                    repo_id,
                    CHECKPOINT_REPO_PATH,
                    {
                        "version": CHECKPOINT_VERSION,
                        "metadata": checkpoint_metadata,
                        "completed_sources": sorted(completed_sources),
                        "results": results,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
    finally:
        manager.shutdown()

    results["realized_total_tokens"] = sum(
        int(source_result["realized_tokens"])
        for source_result in results["sources"].values()
    )
    upload_json(api, repo_id, f"metadata/{run_id}.json", results)
    upload_json(
        api, repo_id, "metadata/mixture.json", {"sources": sources, **budget_plan}
    )
    LOG.info("realized_total_tokens=%s", f"{results['realized_total_tokens']:,}")
    return results


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    args = parse_args()

    try:
        run(args=args)
        return 0
    except LocalStorageSafetyStop as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# python dataset/stream.py --sources "config/source.json" --repo-id "SamLegion888/SWE-test-001" --total-tokens 2000000 --shard-size 512 --num-workers 1 --num-tasks 1
