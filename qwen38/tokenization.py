from pathlib import Path
import json
import re
import keyword
import tokenize as py_tokenize
from tokenizers import Tokenizer, pre_tokenizers, decoders, Regex, normalizers, trainers
from tokenizers.models import BPE


class Qwen38Tokenizer:
    def __init__(
        self,
        vocab=None,
        merges=None,
        add_prefix_space=None,
    ) -> None:
        self.add_prefix_space = (
            add_prefix_space if add_prefix_space is not None else False
        )
        self._vocab = (
            vocab
            if vocab is not None
            else {
                "<|endoftext|>": 0,
            }
        )
        self._merges = merges or []
        self.bpe = BPE(
            vocab=self._vocab,
            merges=self._merges,
            dropout=None,
            unk_token=None,
            continuing_subword_prefix="",
            end_of_word_suffix="",
            fuse_unk=False,
            byte_fallback=False,
        )
        self._tokenizer = Tokenizer(self.bpe)
        self._tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer.normalizer = normalizers.NFC()

        self.python_keywords = list(keyword.kwlist) + list(keyword.softkwlist)
        self.python_keywords.sort(key=len, reverse=True)
        self.python_operators = list(py_tokenize.EXACT_TOKEN_TYPES.keys())
        self.python_operators.sort(key=len, reverse=True)

        python_token_pattern = (
            r"\b(?:"
            + "|".join(map(re.escape, self.python_keywords))
            + r")\b"
            + "|"
            + "|".join(map(re.escape, self.python_operators))
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    Regex(python_token_pattern),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space,
                    use_regex=False,
                ),
            ]
        )

    @classmethod
    def train(
        cls,
        texts,
        vocab_size=16_000,
        min_frequency=2,
        special_tokens=None,
        add_prefix_space=False,
    ):
        tokenizer = cls(add_prefix_space=add_prefix_space)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens or ["<|endoftext|>"],
            show_progress=False,
        )
        tokenizer._tokenizer.train_from_iterator(texts, trainer=trainer)

        print(f"Training finished... Adding lexical tokens")
        python_keywords = list(keyword.kwlist) + list(keyword.softkwlist)
        python_operators = list(py_tokenize.EXACT_TOKEN_TYPES.keys())
        lexical_tokens = list(dict.fromkeys(python_keywords + python_operators))
        tokenizer._tokenizer.add_tokens(lexical_tokens)
        return tokenizer

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self._tokenizer.save(str(directory / "tokenizer.json"))
        self._tokenizer.model.save(str(directory))

        tokenizer_config = {
            "add_prefix_space": self.add_prefix_space,
            "clean_up_tokenization_spaces": False,
            "model_max_length": 4096,
            "tokenizer_class": "Qwen38Tokenizer",
            "bos_token": None,
            "eos_token": "<|endoftext|>",
            "unk_token": None,
            "pad_token": None,
        }
        special_tokens_map = {"eos_token": "<|endoftext|>"}

        with (directory / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        with (directory / "special_tokens_map.json").open("w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

    def encode(self, text):
        return self._tokenizer.encode(text)

    def batch_encode(self, text):
        return self._tokenizer.encode_batch(text)


# if __name__ == "__main__":
#    code = '''
# You are given a list of `n` tasks, each represented as a tuple `(start, end)`, indicating the start and end times of the task. The tasks are sorted by their start times. Your goal is to determine the maximum number of non-overlapping tasks that can be selected. Two tasks are considered non-overlapping if the start time of one task is greater than or equal to the end time of the other.

# **Input:**
# - An integer `n` representing the number of tasks.
# - A list of `n` tuples, where each tuple `(start, end)` represents the start and end times of a task.

# **Output:**
# - An integer representing the maximum number of non-overlapping tasks that can be selected.

# **Constraints:**
# - `1 <= n <= 10^5`
# - `0 <= start < end <= 10^9`

# **Sample Input:**
# ```
# 3
# 1 3
# 2 5
# 4 6
# ```

# **Sample Output:**
# ```
# 2
# ```

# ```python
# def max_non_overlapping_tasks(tasks):
#    """
#    Returns the maximum number of non-overlapping tasks that can be selected from a list of tasks.

#    :param tasks: List of tuples, where each tuple (start, end) represents the start and end times of a task.
#    :return: Integer representing the maximum number of non-overlapping tasks.
#    """
#    if not tasks:
#        return 0

#    # initialize count with 1
#    count = 1
#    last_end = tasks[0][1]

#    for i in range(1, len(tasks)):
#        current_start, current_end = tasks[i]
#        if current_start >= last_end:
#            count += 1
#            last_end = current_end

#    return count
# ```
#'''

#    t = Qwen38Tokenizer()
#    t.tokenize(code)
