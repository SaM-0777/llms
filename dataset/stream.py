from dataclasses import dataclass
import hashlib
import io
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import sys
from array import array
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm
from transformers import AutoTokenizer
import tyro


@dataclass
class Config:
    config: Path = Path("config/mixture.json")
    """Path to the dataset mixture configuration JSON file."""
    validate: bool = False
    """Validate the repos and fields"""
    tokenizer: Path = Path("tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended")
    """Path to the local tokenizer."""
    repo: str = "SamLegion888/SWE_3.5_S"
    """Hugging Face dataset repository in Username/repo_name format."""
    shard_tokens: int = 1000000000  # approximately 4 GB of uint32 token IDs
    """Target number of tokens per shard."""
    num_workers: int = os.cpu_count() or 4
    """Number of workers available for data processing."""
    batch_size: int = 8192
    """Number of documents tokenized together per batch."""
    upload_workers: int = 2
    """Number of concurrent shard uploads."""
    max_pending_uploads: int = 4
    """Maximum completed shards queued or uploading before processing applies backpressure."""
    checkpoint_batches: int = 8
    """Save a durable checkpoint after this many batches."""


def print_result(result: dict[str, Any]) -> None:
    print(f"[{result['status']}] {result['id']} | {result['repo']}")
    if result["error"]:
        print(f"  Error: {result['error']}")


def print_summary(results: list[dict[str, Any]]) -> None:
    succeeded = [r for r in results if r["status"] == "SUCCESS"]
    failed = [r for r in results if r["status"] == "FAILED"]

    print("\nValidation Summary")
    print(f"Total sources: {len(results)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")

    if succeeded:
        print("\nSuccessful repositories:")
        for result in succeeded:
            print(f"  - {result['id']} ({result['repo']})")

    if failed:
        print("\nFailed repositories:")
        for result in failed:
            print(f"  - {result['id']} ({result['repo']}): {result['error']}")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hf_login() -> None:
    api = HfApi()
    try:
        api.whoami()
    except:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN environment variable is required.")
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            raise RuntimeError("Hugging Face login failed. Check your token.") from e


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temp, path)


def load_tokenizer(path: Path) -> AutoTokenizer:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {path}")
    return AutoTokenizer.from_pretrained(path, use_fast=True)


class DataRepo:
    def __init__(
        self,
        config: dict[str, Any],
        tokenizer: AutoTokenizer,
        repo: str,
        shard_tokens: int,
        num_workers: int,
        batch_size: int = 256,
        upload_workers: int = 2,
        max_pending_uploads: int = 4,
        checkpoint_batches: int = 8,
        state_path: Path = Path(".mixture_state.json"),
        work_dir: Path = Path(".mixture_shards"),
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.repo = repo
        self.shard_tokens = shard_tokens
        self.num_workers = max(1, num_workers)
        os.environ.setdefault("RAYON_NUM_THREADS", str(self.num_workers))
        self.batch_size = max(1, batch_size)
        self.upload_workers = max(1, upload_workers)
        self.max_pending_uploads = max(self.upload_workers, max_pending_uploads)
        self.checkpoint_batches = max(1, checkpoint_batches)
        self._writer = None
        self.state_path = state_path
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()
        self.results: list[dict[str, Any]] = []
        self.upload_executor = ThreadPoolExecutor(max_workers=self.upload_workers)
        self.source_executor = ThreadPoolExecutor(max_workers=1)
        self.upload_futures: dict[int, Future] = {}
        self.prefetched_source_key: str | None = None
        self.prefetched_source_future: Future | None = None
        self.config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "config_hash": self.config_hash,
                "category_index": 0,
                "source_index": 0,
                "source_rows_processed": 0,
                "source_tokens_processed": 0,
                "total_tokens": 0,
                "total_rows": 0,
                "total_chars": 0,
                "shard_index": 0,
                "current_shard": None,
                "pending_shard": None,
                "pending_uploads": [],
                "uploaded_shards": [],
                "revisions": {},
                "source_stats": {},
                "category_stats": {},
                "started_at": self.started_at,
                "complete": False,
            }

        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state["config_hash"] != self.config_hash:
            raise ValueError("The mixture configuration differs from the checkpoint.")
        state.setdefault("pending_uploads", [])
        legacy_pending = state.get("pending_shard")
        if legacy_pending is not None:
            if all(
                item["index"] != legacy_pending["index"]
                for item in state["pending_uploads"]
            ):
                state["pending_uploads"].insert(0, legacy_pending)
            state["pending_shard"] = None
        return state

    def _save_state(self) -> None:
        save_json(self.state_path, self.state)

    def _source_key(self, category: dict[str, Any], source: dict[str, Any]) -> str:
        return f"{category['name']}::{source.get('id', source['repo'])}"

    def _get_revision(self, key: str, source: dict[str, Any]) -> str:
        revisions = self.state["revisions"]
        if key not in revisions:
            revision = source.get("revision")
            if not revision:
                revision = self.api.dataset_info(source["repo"]).sha
            revisions[key] = revision
            self._save_state()
        return revisions[key]

    def _format_row(self, row: dict[str, Any], source: dict[str, Any]) -> str:
        fields = list(source.get("fields", []))
        if source.get("field"):
            fields.append(source["field"])
        fields = list(dict.fromkeys(fields))

        if not fields:
            raise ValueError(f"No fields configured for {source['repo']}.")

        missing = set(fields) - set(row)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")

        values = [str(row[field]) for field in fields if row[field] is not None]
        return "\n".join(values)

    def _encode_batch(self, texts: list[str]) -> list[list[int]]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError(
                "The tokenizer must define eos_token_id to terminate each row."
            )
        return [list(ids) + [eos_id] for ids in encoded["input_ids"]]

    def _ensure_shard(self, category: dict[str, Any], source: dict[str, Any]) -> None:
        if self.state["current_shard"] is not None:
            return

        index = self.state["shard_index"]
        path = self.work_dir / f"shard-{index:08d}.bin"
        key = self._source_key(category, source)

        self.state["current_shard"] = {
            "index": index,
            "local_path": str(path),
            "path_in_repo": f"shards/shard-{index:08d}.bin",
            "tokens": 0,
            "bytes": 0,
            "rows": 0,
            "source_key": key,
            "repo": source["repo"],
            "config": source.get("config"),
            "split": source["split"],
            "category": category["name"],
        }

        path.write_bytes(b"")
        self._writer = path.open("ab", buffering=4 * 1024 * 1024)
        self._save_state()

    def _write_row(
        self,
        token_ids: list[int],
        text: str,
        category: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        self._ensure_shard(category, source)
        shard = self.state["current_shard"]
        path = Path(shard["local_path"])

        token_array = array("I", token_ids)
        if token_array.itemsize != 4:
            raise RuntimeError(
                "This platform does not support 32-bit unsigned integers."
            )
        if sys.byteorder != "little":
            token_array.byteswap()

        if self._writer is None or self._writer.closed:
            self._writer = path.open("ab", buffering=4 * 1024 * 1024)
        self._writer.write(token_array.tobytes())

        token_count = len(token_ids)
        char_count = len(text)
        key = shard["source_key"]
        category_name = category["name"]

        shard["tokens"] += token_count
        shard["bytes"] += token_count * 4
        shard["rows"] += 1

        source_stats = self.state["source_stats"].setdefault(
            key,
            {
                "id": source.get("id", source["repo"]),
                "repo": source["repo"],
                "config": source.get("config"),
                "split": source["split"],
                "rows": 0,
                "tokens": 0,
                "characters": 0,
            },
        )
        source_stats["rows"] += 1
        source_stats["tokens"] += token_count
        source_stats["characters"] += char_count

        category_stats = self.state["category_stats"].setdefault(
            category_name, {"rows": 0, "tokens": 0, "characters": 0}
        )
        category_stats["rows"] += 1
        category_stats["tokens"] += token_count
        category_stats["characters"] += char_count

        self.state["source_rows_processed"] += 1
        self.state["source_tokens_processed"] += token_count
        self.state["total_rows"] += 1
        self.state["total_tokens"] += token_count
        self.state["total_chars"] += char_count

    def _checkpoint(self, force: bool = False) -> None:
        if self._writer is not None and not self._writer.closed:
            self._writer.flush()
            os.fsync(self._writer.fileno())
        elif self.state["current_shard"] is not None:
            path = Path(self.state["current_shard"]["local_path"])
            if path.exists():
                with path.open("rb") as f:
                    os.fsync(f.fileno())
        self._save_state()

    def _upload_shard(self, shard: dict[str, Any]) -> None:
        HfApi().upload_file(
            path_or_fileobj=shard["local_path"],
            path_in_repo=shard["path_in_repo"],
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=f"Upload shard {shard['index']}",
        )

    def _schedule_uploads(self) -> None:
        for shard in self.state["pending_uploads"]:
            if len(self.upload_futures) >= self.upload_workers:
                break
            if shard["index"] not in self.upload_futures:
                self.upload_futures[shard["index"]] = self.upload_executor.submit(
                    self._upload_shard, shard
                )

    def _complete_upload(self, shard_index: int) -> None:
        shard = next(
            (
                item
                for item in self.state["pending_uploads"]
                if item["index"] == shard_index
            ),
            None,
        )
        if shard is None:
            return
        if all(item["index"] != shard_index for item in self.state["uploaded_shards"]):
            self.state["uploaded_shards"].append(
                {
                    "index": shard["index"],
                    "path": shard["path_in_repo"],
                    "tokens": shard["tokens"],
                    "rows": shard["rows"],
                    "source": shard["source_key"],
                    "category": shard["category"],
                }
            )
        Path(shard["local_path"]).unlink(missing_ok=True)
        self.state["pending_uploads"] = [
            item
            for item in self.state["pending_uploads"]
            if item["index"] != shard_index
        ]
        self._save_state()

    def _harvest_uploads(self, wait_for_one: bool = False) -> None:
        if wait_for_one and self.upload_futures:
            done, _ = wait(
                list(self.upload_futures.values()), return_when=FIRST_COMPLETED
            )
            done_ids = {id(future) for future in done}
        else:
            done_ids = {
                id(future) for future in self.upload_futures.values() if future.done()
            }
        for shard_index, future in list(self.upload_futures.items()):
            if id(future) not in done_ids:
                continue
            future.result()
            del self.upload_futures[shard_index]
            self._complete_upload(shard_index)
        self._schedule_uploads()

    def _wait_for_upload(self) -> None:
        self._schedule_uploads()
        while self.state["pending_uploads"] or self.upload_futures:
            if not self.upload_futures:
                self._schedule_uploads()
                if not self.upload_futures:
                    raise RuntimeError(
                        "Pending shard uploads exist without schedulable upload tasks."
                    )
            self._harvest_uploads(wait_for_one=True)

    def _finalize_shard(self) -> None:
        shard = self.state["current_shard"]
        if shard is None:
            return
        self._checkpoint(force=True)
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.state["current_shard"] = None
        self.state["pending_uploads"].append(shard)
        self.state["shard_index"] += 1
        self._save_state()
        self._schedule_uploads()
        while len(self.state["pending_uploads"]) >= self.max_pending_uploads:
            self._harvest_uploads(wait_for_one=True)

    def _dataset_kwargs(self, source: dict[str, Any], revision: str) -> dict[str, Any]:
        kwargs = {
            "path": source["repo"],
            "split": source["split"],
            "streaming": True,
            "revision": revision,
        }
        if source.get("config"):
            kwargs["name"] = source["config"]
        return kwargs

    def _resolve_and_load_source(
        self, source: dict[str, Any], existing_revision: str | None
    ):
        revision = existing_revision or source.get("revision")
        if not revision:
            revision = HfApi().dataset_info(source["repo"]).sha
        dataset = load_dataset(**self._dataset_kwargs(source, revision))
        return revision, dataset

    def _prefetch_next_source(self) -> None:
        if self.prefetched_source_future is not None:
            return
        categories = self.config["categories"]
        ci = self.state["category_index"]
        si = self.state["source_index"] + 1
        if si >= len(categories[ci]["sources"]):
            ci += 1
            si = 0
        if ci >= len(categories):
            return
        category = categories[ci]
        source = category["sources"][si]
        key = self._source_key(category, source)
        revision = self.state["revisions"].get(key)
        self.prefetched_source_key = key
        self.prefetched_source_future = self.source_executor.submit(
            self._resolve_and_load_source, source, revision
        )

    def _restore_current_shard(self) -> None:
        shard = self.state["current_shard"]
        if shard is None:
            return

        path = Path(shard["local_path"])
        if not path.exists():
            if shard["bytes"]:
                raise FileNotFoundError(f"Checkpoint shard is missing: {path}")
            path.touch()

        with path.open("r+b") as f:
            f.truncate(shard["bytes"])
            f.flush()
            os.fsync(f.fileno())
        self._writer = path.open("ab", buffering=4 * 1024 * 1024)

    def _advance_source(self) -> None:
        self.state["source_rows_processed"] = 0
        self.state["source_tokens_processed"] = 0
        self.state["source_index"] += 1

        categories = self.config["categories"]
        if self.state["source_index"] >= len(
            categories[self.state["category_index"]]["sources"]
        ):
            self.state["category_index"] += 1
            self.state["source_index"] = 0

        self._save_state()

    def _source_target(self, category: dict[str, Any], source: dict[str, Any]) -> float:
        sources = category["sources"]
        total_weight = sum(item.get("weight", 1) for item in sources)
        return category["target_tokens"] * source.get("weight", 1) / total_weight

    def _summary(self) -> dict[str, Any]:
        return {
            "name": self.config.get("name"),
            "repo": self.repo,
            "total_tokens": self.state["total_tokens"],
            "total_rows": self.state["total_rows"],
            "total_characters": self.state["total_chars"],
            "shards_pushed": len(self.state["uploaded_shards"]),
            "categories": self.state["category_stats"],
            "sources": self.state["source_stats"],
            "started_at": self.state["started_at"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _metadata(self) -> dict[str, Any]:
        return {
            "name": self.config.get("name"),
            "repo": self.repo,
            "format": "raw_token_ids",
            "token_dtype": "uint32",
            "byte_order": "little",
            "tokens_per_row_terminator": self.tokenizer.eos_token_id,
            "tokenizer": {
                "name_or_path": self.tokenizer.name_or_path,
                "vocab_size": len(self.tokenizer),
            },
            "target_shard_tokens": self.shard_tokens,
            "actual_total_tokens": self.state["total_tokens"],
            "categories": self.state["category_stats"],
            "sources": {
                key: {
                    **stats,
                    "revision": self.state["revisions"].get(key),
                }
                for key, stats in self.state["source_stats"].items()
            },
            "shards": self.state["uploaded_shards"],
        }

    def _upload_json(self, filename: str, data: dict[str, Any]) -> None:
        content = json.dumps(data, indent=2).encode("utf-8")
        self.api.upload_file(
            path_or_fileobj=io.BytesIO(content),
            path_in_repo=filename,
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=f"Upload {filename}",
        )

    def stream_and_upload(self) -> dict[str, Any]:
        self.api.create_repo(
            repo_id=self.repo, repo_type="dataset", private=False, exist_ok=True
        )
        self._restore_current_shard()
        self._schedule_uploads()

        total_target = sum(
            category["target_tokens"] for category in self.config["categories"]
        )
        start_time = time.monotonic()
        last_checkpoint_batch = 0
        batch_count = 0
        last_progress_time = start_time
        last_progress_tokens = self.state["total_tokens"]
        last_progress_rows = self.state["total_rows"]

        try:
            with tqdm(
                total=total_target,
                initial=self.state["total_tokens"],
                desc="Tokenizing",
                unit="tok",
                mininterval=1.0,
                smoothing=0.1,
            ) as progress:
                while self.state["category_index"] < len(self.config["categories"]):
                    category = self.config["categories"][self.state["category_index"]]
                    source = category["sources"][self.state["source_index"]]
                    key = self._source_key(category, source)
                    target = self._source_target(category, source)

                    if self.state["source_tokens_processed"] >= target:
                        self._finalize_shard()
                        self._advance_source()
                        continue

                    if (
                        self.prefetched_source_future is not None
                        and self.prefetched_source_key == key
                    ):
                        revision, dataset = self.prefetched_source_future.result()
                        self.state["revisions"][key] = revision
                        self._save_state()
                        self.prefetched_source_future = None
                        self.prefetched_source_key = None
                    else:
                        revision = self._get_revision(key, source)
                        dataset = load_dataset(**self._dataset_kwargs(source, revision))

                    iterator = iter(dataset)
                    for _ in range(self.state["source_rows_processed"]):
                        try:
                            next(iterator)
                        except StopIteration as e:
                            raise RuntimeError(
                                f"Could not resume source {key} at the saved row position."
                            ) from e

                    self._prefetch_next_source()
                    exhausted = False
                    source_done = False

                    while not source_done:
                        rows = []
                        for _ in range(self.batch_size):
                            try:
                                rows.append(next(iterator))
                            except StopIteration:
                                exhausted = True
                                break
                        if not rows:
                            source_done = True
                            break

                        texts = [self._format_row(row, source) for row in rows]
                        encoded_rows = self._encode_batch(texts)
                        for text, token_ids in zip(texts, encoded_rows):
                            if self.state["source_tokens_processed"] >= target:
                                source_done = True
                                break
                            self._write_row(token_ids, text, category, source)
                            token_count = len(token_ids)
                            progress.update(token_count)
                            self._harvest_uploads()

                            shard = self.state["current_shard"]
                            if shard and shard["tokens"] >= self.shard_tokens:
                                self._finalize_shard()

                            if self.state["source_tokens_processed"] >= target:
                                source_done = True
                                break

                        batch_count += 1
                        now = time.monotonic()
                        interval = max(now - last_progress_time, 1e-6)
                        interval_tokens = (
                            self.state["total_tokens"] - last_progress_tokens
                        )
                        interval_rows = self.state["total_rows"] - last_progress_rows
                        progress.set_postfix(
                            {
                                "tok/s": f"{interval_tokens / interval:,.0f}",
                                "docs/s": f"{interval_rows / interval:,.1f}",
                                "shards": len(self.state["uploaded_shards"]),
                                "upload_q": len(self.state["pending_uploads"]),
                                "source": source.get("id", source["repo"]),
                            },
                            refresh=False,
                        )
                        last_progress_time = now
                        last_progress_tokens = self.state["total_tokens"]
                        last_progress_rows = self.state["total_rows"]

                        if (
                            batch_count - last_checkpoint_batch
                            >= self.checkpoint_batches
                        ):
                            self._checkpoint()
                            last_checkpoint_batch = batch_count

                        if exhausted:
                            source_done = True
                            break

                    self._finalize_shard()
                    self._advance_source()

            self._wait_for_upload()
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self.state["complete"] = True
            self._save_state()
            summary = self._summary()
            metadata = self._metadata()
            self._upload_json("summary.json", summary)
            self._upload_json("metadata.json", metadata)
            return summary
        finally:
            if self._writer is not None:
                try:
                    self._writer.flush()
                    self._writer.close()
                finally:
                    self._writer = None
            self.upload_executor.shutdown(wait=True)
            self.source_executor.shutdown(wait=True)

    def validate(self, preview: bool = True) -> list[dict[str, Any]]:
        for category in self.config["categories"]:
            for source in category["sources"]:
                result = {
                    "category": category["name"],
                    "id": source.get("id", source["repo"]),
                    "repo": source["repo"],
                    "status": "FAILED",
                    "error": None,
                }
                try:
                    kwargs = {
                        "path": source["repo"],
                        "split": source["split"],
                        "streaming": True,
                    }
                    if source.get("config"):
                        kwargs["name"] = source["config"]

                    dataset = load_dataset(**kwargs)
                    row = next(iter(dataset), None)
                    features = dataset.features
                    fields = source.get("fields", [])
                    if source.get("field"):
                        fields.append(source["field"])

                    if row is None and not features:
                        raise ValueError(
                            "Split is empty; field validation is unavailable."
                        )

                    available_fields = (
                        set(features.keys())
                        if features
                        else set(row.keys()) if row else set()
                    )
                    missing_fields = set(fields) - available_fields

                    if missing_fields:
                        raise ValueError(f"Missing fields: {sorted(missing_fields)}")

                    result["status"] = "SUCCESS"
                    result["error"] = None

                    if preview:
                        if row is None:
                            print("No rows available for preview.")

                        print(f"\n[{category['name']}] {result['id']}")
                        for field in fields:
                            print(f"\n{field}:\n{row[field]}")

                except Exception as e:
                    result["error"] = f"{type(e).__name__}: {e}"

                self.results.append(result)
                print_result(result)

        print_summary(results=self.results)
        return self.results


def main(args: Config):
    dataset_config = load_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer)

    hf_login()

    data_repo = DataRepo(
        config=dataset_config,
        tokenizer=tokenizer,
        repo=args.repo,
        shard_tokens=args.shard_tokens,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        upload_workers=args.upload_workers,
        max_pending_uploads=args.max_pending_uploads,
        checkpoint_batches=args.checkpoint_batches,
    )

    if args.validate:
        data_repo.validate()
        return

    summary = data_repo.stream_and_upload()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)
