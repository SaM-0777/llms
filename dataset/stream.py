from dataclasses import dataclass
import hashlib
import io
import json
import os
import sys
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
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
    shard_tokens: int = 10_000_000_000
    """Target number of tokens per shard."""
    num_workers: int = os.cpu_count() or 4
    """Number of workers to process data"""


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
    return AutoTokenizer.from_pretrained(path)


class DataRepo:
    def __init__(
        self,
        config: dict[str, Any],
        tokenizer: AutoTokenizer,
        repo: str,
        shard_tokens: int,
        num_workers: int,
        state_path: Path = Path(".mixture_state.json"),
        work_dir: Path = Path(".mixture_shards"),
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.repo = repo
        self.shard_tokens = shard_tokens
        self.num_workers = num_workers
        self.state_path = state_path
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()
        self.results: list[dict[str, Any]] = []
        self.upload_executor = ThreadPoolExecutor(max_workers=1)
        self.token_executor = ThreadPoolExecutor(max_workers=num_workers)
        self.upload_future: Future | None = None
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

    def _encode_text(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids.append(self.tokenizer.eos_token_id)
        return token_ids

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

        with path.open("ab") as f:
            token_array.tofile(f)

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
        shard = self.state["current_shard"]
        if shard is not None:
            path = Path(shard["local_path"])
            if path.exists():
                with path.open("rb") as f:
                    os.fsync(f.fileno())

        self._save_state()

    def _upload_shard(self, shard: dict[str, Any]) -> None:
        self.api.upload_file(
            path_or_fileobj=shard["local_path"],
            path_in_repo=shard["path_in_repo"],
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=f"Upload shard {shard['index']}",
        )

    def _complete_pending_upload(self) -> None:
        shard = self.state["pending_shard"]
        if shard is None:
            return

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
        self.state["pending_shard"] = None
        self._save_state()

    def _harvest_upload(self) -> None:
        if self.upload_future is None or not self.upload_future.done():
            return

        self.upload_future.result()
        self._checkpoint(force=True)
        self._complete_pending_upload()
        self.upload_future = None

    def _wait_for_upload(self) -> None:
        if self.upload_future is not None:
            self.upload_future.result()
            self._checkpoint(force=True)
            self._complete_pending_upload()
            self.upload_future = None
        elif self.state["pending_shard"] is not None:
            self._upload_shard(self.state["pending_shard"])
            self._complete_pending_upload()

    def _finalize_shard(self) -> None:
        shard = self.state["current_shard"]
        if shard is None:
            return

        self._checkpoint(force=True)
        self._wait_for_upload()

        self.state["current_shard"] = None
        self.state["pending_shard"] = shard
        self.state["shard_index"] += 1
        self._save_state()

        self.upload_future = self.upload_executor.submit(self._upload_shard, shard)

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
            repo_id=self.repo,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )

        self._wait_for_upload()
        self._restore_current_shard()

        total_target = sum(
            category["target_tokens"] for category in self.config["categories"]
        )

        with tqdm(
            total=total_target,
            initial=self.state["total_tokens"],
            desc="Tokens processed",
            unit="tok",
        ) as progress:
            while self.state["category_index"] < len(self.config["categories"]):
                category = self.config["categories"][self.state["category_index"]]
                source = category["sources"][self.state["source_index"]]
                key = self._source_key(category, source)
                target = self._source_target(category, source)

                revision = self._get_revision(key, source)
                if self.state["source_tokens_processed"] >= target:
                    self._finalize_shard()
                    self._advance_source()
                    continue

                kwargs = {
                    "path": source["repo"],
                    "split": source["split"],
                    "streaming": True,
                    "revision": revision,
                }
                if source.get("config"):
                    kwargs["name"] = source["config"]

                dataset = load_dataset(**kwargs)
                iterator = iter(dataset)

                for _ in range(self.state["source_rows_processed"]):
                    try:
                        next(iterator)
                    except StopIteration as e:
                        raise RuntimeError(
                            f"Could not resume source {key} at the saved row position."
                        ) from e

                exhausted = False
                source_done = False

                rows = []
                while not source_done:
                    rows = []
                    for _ in range(self.num_workers):
                        try:
                            rows.append(next(iterator))
                        except StopIteration:
                            exhausted = True
                            break

                    if not rows:
                        break

                texts = [self._format_row(row, source) for row in rows]
                encoded_rows = list(self.token_executor.map(self._encode_text, texts))

                for text, token_ids in zip(texts, encoded_rows):
                    if self.state["source_tokens_processed"] >= target:
                        source_done = True
                        break

                    self._write_row(token_ids, text, category, source)
                    progress.update(len(token_ids))
                    progress.set_postfix(
                        shards_pushed=len(self.state["uploaded_shards"])
                    )
                    self._harvest_upload()

                    if self.state["source_rows_processed"] % 1000 == 0:
                        self._checkpoint()

                    shard = self.state["current_shard"]
                    if shard and shard["tokens"] >= self.shard_tokens:
                        self._finalize_shard()

                    if self.state["source_tokens_processed"] >= target:
                        source_done = True
                        break

                if exhausted:
                    break

        self._wait_for_upload()
        self.state["complete"] = True
        self._save_state()

        summary = self._summary()
        metadata = self._metadata()
        self._upload_json("summary.json", summary)
        self._upload_json("metadata.json", metadata)

        self.token_executor.shutdown(wait=True)
        self.upload_executor.shutdown(wait=True)
        return summary

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
    )

    if args.validate:
        data_repo.validate()


if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)
