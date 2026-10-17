from __future__ import annotations

import unittest
import json
import shutil
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from tokenizer.lexical_matching import build_lexical_regex


ROOT = Path(__file__).resolve().parents[1]
EXTENDED_TOKENIZER = ROOT / "tokenizer/NVIDIA-Nemotron-3-Super-120B-A12B-vocab-extended"


class LexicalMatcherUnitTests(unittest.TestCase):
    def test_regex_orders_overlapping_operators_longest_first(self):
        pattern = build_lexical_regex(["<", "<<", "<<="])
        self.assertLess(pattern.index("<<="), pattern.index("|<<"))
        self.assertLess(pattern.index("|<<"), pattern.rindex("|<"))

    def test_latex_control_word_has_a_letter_boundary(self):
        pattern = build_lexical_regex([r"\alpha"])
        self.assertTrue(pattern.endswith(r")"))
        self.assertIn(r"\b", pattern)


@unittest.skipUnless(EXTENDED_TOKENIZER.exists(), "extended tokenizer is unavailable")
class LexicalMatcherIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extended = AutoTokenizer.from_pretrained(
            EXTENDED_TOKENIZER,
            trust_remote_code=True,
        )

        report = json.loads(
            (EXTENDED_TOKENIZER / "extension_report.json").read_text(
                encoding="utf-8"
            )
        )
        cls._temp_dir = tempfile.TemporaryDirectory(prefix="nemotron-base-")
        base_path = Path(cls._temp_dir.name) / "base"
        shutil.copytree(EXTENDED_TOKENIZER, base_path)

        tokenizer_json_path = base_path / "tokenizer.json"
        tokenizer_json = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        original_vocab_size = report["original_vocabulary_size"]
        tokenizer_json["added_tokens"] = [
            entry
            for entry in tokenizer_json["added_tokens"]
            if entry["id"] < original_vocab_size
        ]
        tokenizer_json["pre_tokenizer"] = report["lexical_matching"][
            "original_pre_tokenizer"
        ]
        tokenizer_json_path.write_text(
            json.dumps(tokenizer_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        cls.base = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    def _tokens(self, tokenizer, text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.convert_ids_to_tokens(ids)

    def test_unmatched_conversation_keeps_original_pipeline(self):
        text = "Hello there, how are you today?"
        self.assertEqual(
            self._tokens(self.extended, text),
            self._tokens(self.base, text),
        )

    def test_identifier_boundary_protects_format(self):
        self.assertEqual(
            self._tokens(self.extended, "format"),
            self._tokens(self.base, "format"),
        )
        self.assertIn("for", self._tokens(self.extended, "for item"))

    def test_longest_match_first_for_overlapping_operators(self):
        self.assertIn("<<=", self._tokens(self.extended, "a<<=b"))

    def test_latex_control_word_does_not_match_inside_longer_command(self):
        self.assertIn(r"\alpha", self._tokens(self.extended, r"\alpha + 1"))
        self.assertNotIn(r"\alpha", self._tokens(self.extended, r"\alphabet"))

    def test_existing_added_latex_words_keep_their_boundaries(self):
        self.assertIn(r"\AAaleph", self._tokens(self.extended, r"\AAaleph"))
        self.assertNotIn(
            r"\AA",
            self._tokens(self.extended, r"\AAaleph"),
        )


if __name__ == "__main__":
    unittest.main()
