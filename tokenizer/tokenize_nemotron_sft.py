import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from datasets import get_dataset_split_names, load_dataset
from transformers import AutoTokenizer


DATASET_ID = "nvidia/Nemotron-SFT-SWE-v3.5"
TOKENIZER_PATH = "tokenizer/Qwen35-4B-vocab-extended"
#TOKENIZER_PATH = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = Path("data/Nemotron-SFT-SWE-v3.5/qwen35-extended")
MAX_SEQ_LENGTH = 4096
TEST_TRAIN_RATIO = 0.1
SEED = 42


@dataclass
class Args:
    dataset: str = DATASET_ID
    tokenizer: str = TOKENIZER_PATH


def process_example(example):
    for message in example.get("messages", []):
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = json.loads(arguments)
    return example


def main(args: Args):
    print(f"Using {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    splits = get_dataset_split_names(args.dataset)

    if "test" in splits:
        dataset = load_dataset(args.dataset, split="train", streaming=True)
    else:
        dataset = load_dataset(args.dataset, split="train")
        dataset = dataset.shuffle(seed=SEED).train_test_split(
            test_size=TEST_TRAIN_RATIO,
            seed=SEED,
        )["train"]

    dataset = dataset.map(process_example)
    dtype = np.uint16 if len(tokenizer) <= 65536 else np.uint32

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token_path = OUTPUT_DIR / "tokens.bin"
    lengths_path = OUTPUT_DIR / "lengths.bin"
    metadata_path = OUTPUT_DIR / "metadata.json"

    examples = 0
    tokens = 0

    with token_path.open("wb") as token_file, lengths_path.open("wb") as lengths_file:
        for example in dataset:
            tools = example.get("tools")
            if isinstance(tools, str):
                tools = json.loads(tools)
            ids = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
                tools=tools,
                **example.get("chat_template_kwargs", {}),
            )[:MAX_SEQ_LENGTH]
            np.asarray(ids, dtype=dtype).tofile(token_file)
            np.asarray([len(ids)], dtype=np.uint64).tofile(lengths_file)
            examples += 1
            tokens += len(ids)

    metadata = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "split": "train",
        "test_train_ratio": TEST_TRAIN_RATIO,
        "seed": SEED,
        "max_seq_length": MAX_SEQ_LENGTH,
        "packing": False,
        "examples": examples,
        "tokens": tokens,
        "dtype": np.dtype(dtype).name,
        "token_file": str(token_path),
        "lengths_file": str(lengths_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"examples: {examples:,}")
    print(f"tokens:   {tokens:,}")
    print(f"token_file: {token_path}")
    print(f"lengths:  {lengths_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
