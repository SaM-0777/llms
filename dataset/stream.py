#!/usr/bin/env python3
"""Stream Hugging Face datasets into tokenizer-ID binary shards.

The pipeline keeps input records remote, tokenizes one record at a time, and
writes only the requested training shards plus small control metadata.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import logging
import os
import signal
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

LOG = logging.getLogger("stream_preprocess")
CHECKPOINT_VERSION = 1


class PipelineError(RuntimeError):
    """Raised for a user-actionable preprocessing failure."""


class ConfigurationError(PipelineError):
    """Raised when the mixture or source mapping is invalid."""


class MetadataOnlySourceError(PipelineError):
    """Raised when a raw-code resolver is required for metadata-only data."""


@dataclasses.dataclass
class Document:
    text: str
    source_id: str


@dataclasses.dataclass
class SourceCursor:
    spec: dict[str, Any]
    cache_dir: Path
    code_resolver: Callable[[Mapping[str, Any], Mapping[str, Any]], str | None] | None
    state: dict[str, Any]
    stop_event: Callable[[], bool]

    def __post_init__(self) -> None:
        self.source_id = str(self.spec["id"])
        saved = self.state.setdefault(self.source_id, {})
        self.epoch = int(saved.get("epoch", 0))
        self.position = int(saved.get("position", 0))

    def checkpoint(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}

    def _dataset(self):
        from datasets import DownloadConfig, load_dataset

        repo_id = self.spec["repo"]
        config = self.spec.get("config")
        split = self.spec.get("split", "train")
        kwargs: dict[str, Any] = {
            "path": repo_id,
            "split": split,
            "streaming": True,
            "cache_dir": str(self.cache_dir),
            "trust_remote_code": False,
            "download_config": DownloadConfig(cache_dir=str(self.cache_dir)),
        }
        if config:
            kwargs["name"] = config
        dataset = load_dataset(**kwargs)
        shuffle_buffer = int(self.spec.get("shuffle_buffer", 0))
        if shuffle_buffer > 0:
            dataset = dataset.shuffle(
                seed=int(self.spec.get("shuffle_seed", 0)),
                buffer_size=shuffle_buffer,
            )
        return dataset

    def _format_record(self, record: Mapping[str, Any]) -> str | None:
        source_type = self.spec.get("type", "hf_text")
        if source_type == "hf_metadata":
            if self.code_resolver is None:
                raise MetadataOnlySourceError(
                    f"Source {self.source_id} exposes metadata only. Provide "
                    "--code-resolver module:function that retrieves raw content "
                    "from each metadata record, or configure a text-bearing source "
                    "for this category. Raw content is required for code training."
                )
            return self.code_resolver(record, self.spec)

        formatter = self.spec.get("formatter", "field")
        if formatter == "persona":
            fields = self.spec.get("fields", [])
            parts: list[str] = []
            for field in fields:
                value = record.get(field)
                if value is None or value == "":
                    continue
                if isinstance(value, (list, tuple, dict)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                parts.append(f"{field}: {value}")
            return "\n".join(parts) or None

        field = self.spec.get("field", "text")
        value = record.get(field)
        if value is None:
            raise ConfigurationError(
                f"Source {self.source_id} expected field {field!r}; "
                f"available fields: {sorted(record.keys())}"
            )
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

    def _passes_filter(self, record: Mapping[str, Any], text: str) -> bool:
        rules = self.spec.get("filters", {})
        if not rules:
            return True
        if len(text) < int(rules.get("min_chars", 0)):
            return False
        contains_any = [str(x).lower() for x in rules.get("contains_any", [])]
        if contains_any and not any(x in text.lower() for x in contains_any):
            return False
        source_in = rules.get("source_in")
        if source_in and str(record.get("source")) not in {str(x) for x in source_in}:
            return False
        min_score = rules.get("min_score")
        if min_score is not None and float(record.get("score", 0.0)) < float(min_score):
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        repeat = bool(self.spec.get("repeat", False))
        max_epochs = self.spec.get("max_epochs")
        while True:
            dataset = self._dataset()
            iterator = iter(dataset)
            to_skip = self.position
            if to_skip:
                LOG.info(
                    "Resuming source=%s epoch=%d after %d records",
                    self.source_id,
                    self.epoch,
                    to_skip,
                )
            for _ in range(to_skip):
                if self.stop_event():
                    return
                try:
                    next(iterator)
                except StopIteration:
                    break
            for record in iterator:
                if self.stop_event():
                    return
                self.position += 1
                text = self._format_record(record)
                if text and self._passes_filter(record, text):
                    yield text
            if not repeat:
                return
            self.epoch += 1
            self.position = 0
            if max_epochs is not None and self.epoch >= int(max_epochs):
                return
            LOG.info("Repeating source=%s at epoch=%d", self.source_id, self.epoch)


class CategoryStream:
    def __init__(
        self,
        category: Mapping[str, Any],
        cache_dir: Path,
        source_state: dict[str, Any],
        code_resolver: (
            Callable[[Mapping[str, Any], Mapping[str, Any]], str | None] | None
        ),
        stop_event: Callable[[], bool],
    ) -> None:
        self.category = category
        self.sources = [
            SourceCursor(s, cache_dir, code_resolver, source_state, stop_event)
            for s in category.get("sources", [])
        ]
        self._iterators: list[tuple[SourceCursor, Iterator[str]]] = []
        self._schedule: list[int] = []
        for index, source in enumerate(self.sources):
            weight = max(1, int(source.spec.get("weight", 1)))
            self._schedule.extend([index] * weight)
        if not self._schedule:
            raise ConfigurationError(f"Category {category.get('name')} has no sources")
        self._schedule_pos = 0

    def _ensure_iterators(self) -> None:
        if self._iterators:
            return
        self._iterators = [(source, iter(source)) for source in self.sources]

    def next_document(self) -> Document | None:
        self._ensure_iterators()
        while self._iterators:
            for _ in range(len(self._schedule)):
                source_index = self._schedule[self._schedule_pos % len(self._schedule)]
                self._schedule_pos += 1
                match = next(
                    (
                        item
                        for item in self._iterators
                        if item[0] is self.sources[source_index]
                    ),
                    None,
                )
                if match is None:
                    continue
                source, iterator = match
                try:
                    return Document(next(iterator), source.source_id)
                except StopIteration:
                    self._iterators.remove(match)
                    break
        return None

    def checkpoint(self) -> dict[str, dict[str, int]]:
        return {source.source_id: source.checkpoint() for source in self.sources}


class TokenizerAdapter:
    def __init__(
        self, encode: Callable[[str], list[int]], vocab_size: int, eos_id: int | None
    ):
        self.encode = encode
        self.vocab_size = vocab_size
        self.eos_id = eos_id


def _find_eos_id(tokenizer: Any) -> int | None:
    for token in ("<|endoftext|>", "<|eos|>", "</s>", "<eos>"):
        try:
            value = tokenizer.token_to_id(token)
        except AttributeError:
            value = None
        if value is not None:
            return int(value)
    return None


def load_tokenizer(path: str, eos_override: int | None) -> TokenizerAdapter:
    tokenizer_path = Path(path).expanduser()
    if not tokenizer_path.exists():
        raise ConfigurationError(f"Tokenizer path does not exist: {tokenizer_path}")
    json_path = (
        tokenizer_path
        if tokenizer_path.is_file()
        else tokenizer_path / "tokenizer.json"
    )
    if json_path.exists():
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(json_path))
            eos_id = (
                eos_override if eos_override is not None else _find_eos_id(tokenizer)
            )
            if eos_id is None:
                raise ConfigurationError(
                    "The tokenizer.json has no recognized EOS token. Pass --eos-token-id."
                )
            return TokenizerAdapter(
                lambda text: list(tokenizer.encode(text, add_special_tokens=False).ids),
                int(tokenizer.get_vocab_size()),
                eos_id,
            )
        except ImportError:
            pass

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ConfigurationError(
            "Install tokenizers or transformers to load the local tokenizer."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    eos_id = eos_override
    if eos_id is None:
        eos_value = tokenizer.eos_token_id
        if isinstance(eos_value, (list, tuple)):
            eos_value = eos_value[0] if eos_value else None
        eos_id = int(eos_value) if eos_value is not None else None
    if eos_id is None:
        raise ConfigurationError(
            "The tokenizer has no EOS token. Pass --eos-token-id explicitly."
        )
    return TokenizerAdapter(
        lambda text: list(tokenizer(text, add_special_tokens=False)["input_ids"]),
        int(tokenizer.vocab_size),
        eos_id,
    )


class RemoteShardUploader:
    """Upload completed shards to a Hub dataset repository and release local space."""

    def __init__(self, repo_id: str, prefix: str) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ConfigurationError(
                "Install huggingface_hub to use --upload-repo."
            ) from exc
        self.api = HfApi()
        self.repo_id = repo_id
        self.prefix = prefix.strip("/")
        self.api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        files = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        self.remote_files = set(files)
        self.uploaded_shards = sorted(
            name for name in self.remote_files if self._is_shard_path(name)
        )
        LOG.info(
            "remote repository=%s prefix=%s existing_shards=%d",
            repo_id,
            self.prefix or ".",
            len(self.uploaded_shards),
        )

    def _remote_path(self, filename: str) -> str:
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def _is_shard_path(self, path: str) -> bool:
        return (
            path.startswith(f"{self.prefix}/tokens-")
            if self.prefix
            else path.startswith("tokens-")
        )

    def has_shard(self, shard_index: int) -> bool:
        return self._remote_path(f"tokens-{shard_index:06d}.bin") in self.remote_files

    def upload_shard(self, local_path: Path, token_count: int) -> None:
        remote_path = self._remote_path(local_path.name)
        if remote_path in self.remote_files:
            LOG.info(
                "remote shard already present; releasing local copy: %s", remote_path
            )
        else:
            LOG.info(
                "uploading shard=%s tokens=%d bytes=%d",
                remote_path,
                token_count,
                local_path.stat().st_size,
            )
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Add {local_path.name}",
            )
            self.remote_files.add(remote_path)
            self.uploaded_shards.append(remote_path)
        local_path.unlink(missing_ok=True)

    def upload_json(self, path_in_repo: str, payload: Mapping[str, Any]) -> None:
        from io import BytesIO

        remote_path = self._remote_path(path_in_repo)
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.api.upload_file(
            path_or_fileobj=BytesIO(data),
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Update {path_in_repo}",
        )
        self.remote_files.add(remote_path)


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        dtype: str,
        shard_tokens: int,
        shard_index: int,
        tokens_in_shard: int,
        on_shard_ready: Callable[[Path, int], None] | None = None,
        remote_shard_exists: Callable[[int], bool] | None = None,
    ) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise ConfigurationError(
                "Install numpy to write binary token shards."
            ) from exc
        self.np = np
        if dtype not in {"uint16", "uint32", "uint64"}:
            raise ConfigurationError("--dtype must be uint16, uint32, or uint64")
        self.dtype = dtype
        self.numpy_dtype = {"uint16": "<u2", "uint32": "<u4", "uint64": "<u8"}[dtype]
        self.itemsize = np.dtype(self.numpy_dtype).itemsize
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = int(shard_tokens)
        self.shard_index = int(shard_index)
        self.tokens_in_shard = int(tokens_in_shard)
        self.on_shard_ready = on_shard_ready
        self.remote_shard_exists = remote_shard_exists
        self._file = None
        self._open_current()

    def _path(self) -> Path:
        return self.output_dir / f"tokens-{self.shard_index:06d}.bin"

    def _open_current(self) -> None:
        path = self._path()
        if not path.exists() and self.tokens_in_shard > 0:
            if self.remote_shard_exists and self.remote_shard_exists(self.shard_index):
                LOG.info(
                    "checkpoint shard=%s is already remote; continuing at next shard",
                    path.name,
                )
                self.shard_index += 1
                self.tokens_in_shard = 0
                self._open_current()
                return
            raise PipelineError(
                f"Checkpoint expects local shard {path}; a matching local or remote shard is required"
            )
        mode = "r+b" if path.exists() else "w+b"
        self._file = path.open(mode)
        expected_bytes = self.tokens_in_shard * self.itemsize
        if mode == "r+b":
            self._file.truncate(expected_bytes)
        self._file.seek(expected_bytes)

    def write(self, token_ids: list[int]) -> None:
        offset = 0
        while offset < len(token_ids):
            room = self.shard_tokens - self.tokens_in_shard
            if room == 0:
                self._rotate()
                room = self.shard_tokens
            count = min(room, len(token_ids) - offset)
            values = self.np.asarray(
                token_ids[offset : offset + count], dtype=self.numpy_dtype
            )
            values.tofile(self._file)
            self.tokens_in_shard += count
            offset += count
            if self.tokens_in_shard == self.shard_tokens:
                self._rotate()

    def _rotate(self) -> None:
        self.flush()
        completed_path = self._path()
        completed_tokens = self.tokens_in_shard
        self._file.close()
        if self.on_shard_ready:
            self.on_shard_ready(completed_path, completed_tokens)
        self.shard_index += 1
        self.tokens_in_shard = 0
        self._open_current()

    def flush(self) -> None:
        self._file.flush()
        os.fsync(self._file.fileno())

    def state(self) -> dict[str, int]:
        return {
            "shard_index": self.shard_index,
            "tokens_in_shard": self.tokens_in_shard,
        }

    def close(self) -> None:
        if self._file is None:
            return
        self.flush()
        path = self._path()
        token_count = self.tokens_in_shard
        self._file.close()
        if token_count > 0 and self.on_shard_ready:
            self.on_shard_ready(path, token_count)
        elif token_count == 0 and path.exists():
            path.unlink()
        self._file = None


class InMemoryDeduper:
    def __init__(self, mode: str, max_items: int) -> None:
        self.mode = mode
        self.max_items = max_items
        self.seen: set[bytes] = set()
        self.saturated = False

    def accept(self, text: str) -> bool:
        if self.mode == "none" or self.saturated:
            return True
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if digest in self.seen:
            return False
        if len(self.seen) >= self.max_items:
            self.saturated = True
            LOG.warning(
                "In-memory deduplication limit reached; continuing without new hashes"
            )
            return True
        self.seen.add(digest)
        return True


def load_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ConfigurationError("Callable must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ConfigurationError(f"Callable is unavailable: {spec}")
    return function


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    categories = config.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ConfigurationError(
            "Configuration must contain a non-empty categories list"
        )
    share_sum = sum(float(category.get("share", 0)) for category in categories)
    if abs(share_sum - 100.0) > 1e-6:
        raise ConfigurationError(f"Category shares sum to {share_sum}, expected 100")
    target_sum = sum(int(category.get("target_tokens", 0)) for category in categories)
    expected = int(config.get("total_tokens", target_sum))
    if target_sum != expected:
        raise ConfigurationError(
            f"Category token budgets sum to {target_sum}, expected {expected}"
        )
    seen_sources: set[str] = set()
    for category in categories:
        for source in category.get("sources", []):
            source_id = str(source.get("id", ""))
            if not source_id or source_id in seen_sources:
                raise ConfigurationError(f"Source ids must be unique: {source_id!r}")
            seen_sources.add(source_id)
            for field in ("repo", "split"):
                if not source.get(field):
                    raise ConfigurationError(f"Source {source_id} needs {field}")


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def build_checkpoint(
    path: Path,
    config: Mapping[str, Any],
    tokenizer_path: str,
    writer: ShardWriter,
    source_state: Mapping[str, Any],
    category_tokens: Mapping[str, int],
    category_docs: Mapping[str, int],
    docs_seen: int,
) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "config_sha256": config_hash(config),
        "tokenizer_path": str(Path(tokenizer_path).expanduser().resolve()),
        "writer": writer.state(),
        "source_state": source_state,
        "category_tokens": category_tokens,
        "category_docs": category_docs,
        "docs_seen": docs_seen,
        "updated_at_unix": time.time(),
    }


def validate_sources(
    config: Mapping[str, Any],
    cache_dir: Path,
    code_resolver: Callable[..., Any] | None,
) -> None:
    stop = lambda: False
    source_state: dict[str, Any] = {}
    checked = 0
    for category in config["categories"]:
        for spec in category["sources"]:
            cursor = SourceCursor(spec, cache_dir, code_resolver, source_state, stop)
            iterator = iter(cursor)
            try:
                text = next(iterator)
            except StopIteration as exc:
                raise PipelineError(f"Source is empty: {spec['id']}") from exc
            if not isinstance(text, str) or not text.strip():
                raise PipelineError(f"Source produced no usable text: {spec['id']}")
            checked += 1
            LOG.info("validated source=%s sample_chars=%d", spec["id"], len(text))
    LOG.info("validated %d configured sources", checked)


def run_pipeline(args: argparse.Namespace) -> int:
    config = load_json(Path(args.config).expanduser())
    validate_config(config)
    output_dir = Path(args.output).expanduser().resolve()
    checkpoint_path = output_dir / "checkpoint.json"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for stale in output_dir.glob("tokens-*.bin"):
            stale.unlink()
        for stale in (checkpoint_path, manifest_path):
            if stale.exists():
                stale.unlink()

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        LOG.warning(
            "received signal %s; finishing the current record and checkpointing", signum
        )

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        with tempfile.TemporaryDirectory(prefix="hf-stream-cache-") as cache_name:
            cache_dir = Path(cache_name)
            for key in (
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "HF_HUB_CACHE",
                "HF_XET_CACHE",
                "TRANSFORMERS_CACHE",
            ):
                os.environ[key] = str(cache_dir / key.lower())
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = load_tokenizer(args.tokenizer, args.eos_token_id)
            if args.dtype == "auto":
                dtype = "uint16" if tokenizer.vocab_size <= 65535 else "uint32"
            else:
                dtype = args.dtype
            max_id = tokenizer.vocab_size - 1
            if dtype == "uint16" and max_id > 65535:
                raise ConfigurationError(
                    "uint16 cannot represent the tokenizer vocabulary"
                )

            if args.validate_only:
                validate_sources(
                    config,
                    cache_dir,
                    load_callable(args.code_resolver) if args.code_resolver else None,
                )
                return 0

            source_state: dict[str, Any] = {}
            category_tokens = {str(c["name"]): 0 for c in config["categories"]}
            category_docs = {str(c["name"]): 0 for c in config["categories"]}
            docs_seen = 0
            writer_state = {"shard_index": 0, "tokens_in_shard": 0}
            if checkpoint_path.exists() and not args.fresh:
                checkpoint = load_json(checkpoint_path)
                if checkpoint.get("version") != CHECKPOINT_VERSION:
                    raise PipelineError(
                        "Checkpoint version is incompatible with this script"
                    )
                if checkpoint.get("config_sha256") != config_hash(config):
                    raise PipelineError(
                        "Checkpoint configuration differs from --config"
                    )
                source_state = dict(checkpoint.get("source_state", {}))
                category_tokens.update(
                    {
                        k: int(v)
                        for k, v in checkpoint.get("category_tokens", {}).items()
                    }
                )
                category_docs.update(
                    {k: int(v) for k, v in checkpoint.get("category_docs", {}).items()}
                )
                docs_seen = int(checkpoint.get("docs_seen", 0))
                writer_state = dict(checkpoint.get("writer", writer_state))
                LOG.info("resuming from %s", checkpoint_path)

            uploader = (
                RemoteShardUploader(args.upload_repo, args.upload_prefix)
                if args.upload_repo
                else None
            )
            if uploader and uploader.uploaded_shards and not checkpoint_path.exists():
                raise PipelineError(
                    "The remote repository already contains token shards. "
                    "Restore the matching checkpoint or choose a new --upload-prefix."
                )
            if uploader and args.fresh and uploader.uploaded_shards:
                raise PipelineError(
                    "--fresh with an existing remote shard set requires a new --upload-prefix."
                )

            checkpoint_hook: dict[str, Callable[[], None] | None] = {"fn": None}

            def on_shard_ready(path: Path, token_count: int) -> None:
                if uploader:
                    uploader.upload_shard(path, token_count)
                if checkpoint_hook["fn"]:
                    checkpoint_hook["fn"]()

            writer = ShardWriter(
                output_dir,
                dtype,
                int(args.shard_tokens),
                int(writer_state["shard_index"]),
                int(writer_state["tokens_in_shard"]),
                on_shard_ready=on_shard_ready if uploader else None,
                remote_shard_exists=uploader.has_shard if uploader else None,
            )
            resolver = load_callable(args.code_resolver) if args.code_resolver else None
            deduper = InMemoryDeduper(args.dedupe, int(args.dedupe_max_items))
            streams = {
                str(category["name"]): CategoryStream(
                    category,
                    cache_dir,
                    source_state,
                    resolver,
                    lambda: stop_requested,
                )
                for category in config["categories"]
            }
            original_targets = {
                str(c["name"]): int(c["target_tokens"]) for c in config["categories"]
            }
            targets = dict(original_targets)
            last_checkpoint_docs = docs_seen
            last_log = time.monotonic()
            last_log_tokens = sum(category_tokens.values())
            started = time.monotonic()

            def persist_checkpoint() -> None:
                for stream in streams.values():
                    source_state.update(stream.checkpoint())
                checkpoint = build_checkpoint(
                    checkpoint_path,
                    config,
                    args.tokenizer,
                    writer,
                    source_state,
                    category_tokens,
                    category_docs,
                    docs_seen,
                )
                write_json_atomic(checkpoint_path, checkpoint)

            checkpoint_hook["fn"] = persist_checkpoint

            while not stop_requested:
                unfinished = [
                    name
                    for name, target in targets.items()
                    if category_tokens[name] < target
                ]
                if not unfinished:
                    break
                category_name = min(
                    unfinished,
                    key=lambda name: category_tokens[name] / max(targets[name], 1),
                )
                document = streams[category_name].next_document()
                if document is None:
                    LOG.warning(
                        "source exhaustion category=%s produced=%d target=%d",
                        category_name,
                        category_tokens[category_name],
                        targets[category_name],
                    )
                    targets[category_name] = category_tokens[category_name]
                    continue
                if not deduper.accept(document.text):
                    continue
                token_ids = tokenizer.encode(document.text)
                if tokenizer.eos_id is not None:
                    token_ids.append(tokenizer.eos_id)
                remaining = targets[category_name] - category_tokens[category_name]
                if len(token_ids) > remaining:
                    if tokenizer.eos_id is not None and remaining >= 1:
                        token_ids = token_ids[: remaining - 1] + [tokenizer.eos_id]
                    else:
                        token_ids = token_ids[:remaining]
                if not token_ids:
                    continue
                writer.write(token_ids)
                category_tokens[category_name] += len(token_ids)
                category_docs[category_name] += 1
                docs_seen += 1

                if docs_seen - last_checkpoint_docs >= int(args.checkpoint_every_docs):
                    writer.flush()
                    persist_checkpoint()
                    last_checkpoint_docs = docs_seen

                now = time.monotonic()
                if now - last_log >= args.log_interval_seconds:
                    total = sum(category_tokens.values())
                    rate = (total - last_log_tokens) / max(now - last_log, 1e-9)
                    overall = total / max(
                        sum(int(c["target_tokens"]) for c in config["categories"]), 1
                    )
                    LOG.info(
                        "progress tokens=%d ratio=%.6f docs=%d rate=%.1f tok/s shard=%d",
                        total,
                        overall,
                        docs_seen,
                        rate,
                        writer.shard_index,
                    )
                    last_log = now
                    last_log_tokens = total

            writer.flush()
            persist_checkpoint()
            writer.close()

            if stop_requested:
                LOG.warning("checkpoint saved after %d documents", docs_seen)
                return 130
            shortfalls = {
                name: original_targets[name] - category_tokens[name]
                for name in original_targets
                if category_tokens[name] < original_targets[name]
            }
            if shortfalls and not args.allow_shortfall:
                raise PipelineError(
                    "Configured sources ended before their budgets. "
                    f"Shortfalls: {shortfalls}. Use --allow-shortfall to finalize intentionally."
                )
            manifest = {
                "format": "little-endian raw token ids",
                "dtype": dtype,
                "eos_token_id": tokenizer.eos_id,
                "tokenizer_path": str(Path(args.tokenizer).expanduser().resolve()),
                "config_sha256": config_hash(config),
                "total_tokens": sum(category_tokens.values()),
                "category_tokens": category_tokens,
                "category_documents": category_docs,
                "shard_tokens": int(args.shard_tokens),
                "upload_repo": args.upload_repo,
                "upload_prefix": args.upload_prefix if args.upload_repo else None,
                "uploaded_shards": len(uploader.uploaded_shards) if uploader else None,
                "completed_at_unix": time.time(),
                "elapsed_seconds": time.monotonic() - started,
            }
            write_json_atomic(manifest_path, manifest)
            if uploader:
                uploader.upload_json("metadata/manifest.json", manifest)
                uploader.upload_json("metadata/mixture.json", config)
            LOG.info(
                "completed total_tokens=%d shards=%s",
                manifest["total_tokens"],
                output_dir,
            )
            return 0
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON mixture configuration")
    parser.add_argument(
        "--tokenizer", required=True, help="Local tokenizer.json or tokenizer directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for token shards and control metadata",
    )
    parser.add_argument(
        "--upload-repo",
        default=None,
        help="Public Hugging Face dataset repository, for example username/pretraining-tokens",
    )
    parser.add_argument(
        "--upload-prefix",
        default="tokens",
        help="Directory inside the Hub dataset repository for uploaded shards",
    )
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument(
        "--dtype", choices=["auto", "uint16", "uint32", "uint64"], default="auto"
    )
    parser.add_argument("--shard-tokens", type=int, default=268435456)
    parser.add_argument("--checkpoint-every-docs", type=int, default=1000)
    parser.add_argument("--log-interval-seconds", type=float, default=30.0)
    parser.add_argument("--dedupe", choices=["none", "sha256"], default="none")
    parser.add_argument("--dedupe-max-items", type=int, default=10_000_000)
    parser.add_argument(
        "--code-resolver",
        default=None,
        help="module:function for metadata-only raw code retrieval",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start with fresh control state in the output directory",
    )
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    if args.checkpoint_every_docs < 1:
        parser.error("--checkpoint-every-docs must be at least 1")
    if args.shard_tokens < 1:
        parser.error("--shard-tokens must be at least 1")
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        LOG.warning("interrupted before a durable checkpoint")
        return 130
    except PipelineError as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


#export HF_TOKEN=hf_your_write_token

#python stream_preprocess.py \
#  --config configs/locked_200b_mixture.json \
#  --tokenizer /path/to/your/tokenizer \
#  --output /path/to/control-directory \
#  --upload-repo YOUR_USER/your-pretraining-tokens \
#  --upload-prefix tokens-v1 \
#  --shard-tokens 268435456 \
#  --dtype auto \
#  --code-resolver my_code_resolver:resolve_code