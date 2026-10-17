"""Boundary-aware lexical matching for the Nemotron tokenizer."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from tokenizers import AddedToken, Regex, pre_tokenizers


_LATEX_CONTROL_WORD_RE = re.compile(r"^\\[A-Za-z]+$")
_REGEX_META = frozenset(r"\\.^$|?*+()[]{}-")


def _rust_regex_escape(value: str) -> str:
    """Escape a literal for the Rust regex engine used by tokenizers."""

    escaped: list[str] = []
    for character in value:
        codepoint = ord(character)
        if character == "\n":
            escaped.append(r"\n")
        elif character == "\r":
            escaped.append(r"\r")
        elif character == "\t":
            escaped.append(r"\t")
        elif character == "\0":
            escaped.append(r"\x00")
        elif character in _REGEX_META:
            escaped.append("\\" + character)
        elif codepoint < 0x20:
            escaped.append(f"\\x{codepoint:02x}")
        else:
            escaped.append(character)
    return "".join(escaped)


def is_latex_control_word(token: str) -> bool:
    """Return whether *token* is a LaTeX control word such as ``\\alpha``."""

    return bool(_LATEX_CONTROL_WORD_RE.fullmatch(token))


def is_word_bounded_lexical(token: str) -> bool:
    """Return whether a lexical string should receive identifier boundaries."""

    if not token or is_latex_control_word(token):
        return False
    first_is_word = token[0].isalnum() or token[0] == "_"
    last_is_word = token[-1].isalnum() or token[-1] == "_"
    return first_is_word and last_is_word


def build_lexical_regex(tokens: Iterable[str]) -> str:
    """Build a longest-match-first regex for registered lexical strings."""

    unique_tokens = {token for token in tokens if token}
    latex_words = sorted(
        (token for token in unique_tokens if is_latex_control_word(token)),
        key=lambda token: (-len(token), token),
    )
    word_tokens = sorted(
        (
            token
            for token in unique_tokens
            if is_word_bounded_lexical(token)
        ),
        key=lambda token: (-len(token), token),
    )
    word_token_set = set(word_tokens)
    latex_word_set = set(latex_words)
    exact_tokens = sorted(
        (
            token
            for token in unique_tokens
            if token not in word_token_set and token not in latex_word_set
        ),
        key=lambda token: (-len(token), token),
    )

    clauses: list[str] = []
    if latex_words:
        clauses.append(
            "(?:"
            + "|".join(_rust_regex_escape(token) for token in latex_words)
            + r")\b"
        )
    if word_tokens:
        clauses.append(
            r"(?: ?\b(?:"
            + "|".join(_rust_regex_escape(token) for token in word_tokens)
            + r")\b)"
        )
    if exact_tokens:
        clauses.append(
            "(?:"
            + "|".join(_rust_regex_escape(token) for token in exact_tokens)
            + ")"
        )

    return "(?:" + "|".join(clauses) + ")" if clauses else r"(?!x)x"


def add_missing_lexical_tokens(tokenizer, tokens: Iterable[str]) -> tuple[list[str], int]:
    """Add missing strings with category-aware matching properties."""

    original_vocab = tokenizer.get_vocab()
    missing_tokens = sorted(
        {token for token in tokens if token and token not in original_vocab},
        key=lambda token: (-len(token), token),
    )
    added_tokens = [
        AddedToken(
            token,
            single_word=is_word_bounded_lexical(token) or is_latex_control_word(token),
            normalized=False,
            special=False,
        )
        for token in missing_tokens
    ]
    return missing_tokens, tokenizer.add_tokens(added_tokens)


def install_lexical_matcher(tokenizer, tokens: Iterable[str]) -> str:
    """Install the lexical layer before the original pre-tokenizer."""

    pattern = build_lexical_regex(tokens)
    lexical_split = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    original_pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [lexical_split, original_pre_tokenizer]
    )
    return pattern


def model_fingerprint(tokenizer) -> str:
    """Hash the serialized BPE model for before/after preservation checks."""

    serialized = json.loads(tokenizer.backend_tokenizer.to_str())
    payload = json.dumps(
        serialized["model"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serialized_pre_tokenizer(tokenizer) -> dict:
    """Return the current pre-tokenizer JSON for reproducibility metadata."""

    return json.loads(tokenizer.backend_tokenizer.to_str())["pre_tokenizer"]


def patch_saved_added_token_metadata(
    tokenizer_json_path,
    tokens: Iterable[str],
    first_added_token_id: int,
) -> int:
    """Apply boundary metadata to lexical entries already saved on disk."""

    path = tokenizer_json_path
    serialized = json.loads(path.read_text(encoding="utf-8"))
    lexical_tokens = set(tokens)
    changed = 0

    for entry in serialized.get("added_tokens", []):
        content = entry.get("content")
        if entry.get("id", -1) < first_added_token_id or content not in lexical_tokens:
            continue

        entry["single_word"] = is_word_bounded_lexical(content) or is_latex_control_word(content)
        entry["lstrip"] = False
        entry["rstrip"] = False
        entry["normalized"] = False
        changed += 1

    path.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return changed
