import os

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class Args:
    """Prepare dataset"""

    """ """
    name: str
    dataset: str | Path
    tokenizer_name: str = "Qwen/Qwen3.8-27B"
    data_dir: str | Path = Path("/data")
    output_dir: str | Path | None = None
    total_batches: int = 1024


def load_tokenizer(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
    )
    return tokenizer


def process(batch, tokenizer):
    texts = batch["text"]
    tokenized = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = tokenized["input_ids"]
    token_ids.append(tokenizer.eos_token_id)
    return {
        "input_ids": token_ids,
        "token_count": len(token_ids),
    }


def prepare_dataset(args: Args):
    name = args.name
    dataset = args.dataset
    data_dir = args.data_dir
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.dataset)

    # check if path exists
    if os.path.exists(dataset):
        print(f"Found local dataset at {dataset}")
    else:
        print(f"No local dataset found. Fetching from hub")

    # load raw dataset corpus
    data = load_dataset(path=str(args.dataset))
    print(f"Found splits: {data.keys()}")

    # process the dataset
    if not os.path.exists(output_dir / Path("train.bin")):
        tokenizer = load_tokenizer(args.tokenizer_name)
        vocab_size = tokenizer.vocab_size
        print(f"Using tokenizer {args.tokenizer_name}")
        print(f"Tokenizer vocab size {tokenizer.vocab_size}")

        if vocab_size < np.iinfo(np.uint16).max + 1:
            dtype = np.uint16
        else:
            dtype = np.uint32

        tokenized = data.map(
            process,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=["text"],
            desc="Tokenizing splits",
            num_proc=os.cpu_count(),
        )

        for split, dset in tokenized.items():
            total_tokens = int(np.sum(dset["token_count"], dtype=np.uint64))
            print(f"Tokenizing split {split} - Total tokens {total_tokens}")

            filename = output_dir / Path(f"{split}.bin")
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(total_tokens,))
            total_batches = args.total_batches

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches,
                    index=batch_idx,
                    contiguous=True,
                )
                arr_batch = np.concatenate(batch["input_ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)

            arr.flush()


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    prepare_dataset(args)
