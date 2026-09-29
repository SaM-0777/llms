from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import IterableDataset


class MemmapDataset(Dataset):
    """A PyTorch Dataset that loads sequences from a memory-mapped .bin file."""

    def __init__(self, data_path, sequence_length, dtype=np.uint16):
        # Memory-map the file
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.sequence_length = sequence_length

    def __len__(self):
        # Return the total number of possible sequences
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get a single sequence and its target
        x = torch.from_numpy(
            self.data[idx : idx + self.sequence_length].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.sequence_length].astype(np.int64)
        )
        return x, y


def token_stream(
    data_path: Path | str,
    sequence_length: int,
):
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    num_samples = ((len(data) - 1)) // sequence_length

    for sample_idx in range(num_samples):
        start = sample_idx * sequence_length

        x = data[start : start + sequence_length].astype(np.int64, copy=True)
        targets = data[start + 1 : start + sequence_length + 1].astype(
            np.int64, copy=True
        )

        yield {
            "x": x,
            "targets": targets,
        }


def create_dataset(
    data_path: str | Path,
    sequence_length: int,
    shuffle_buffer_size: int = 100_000,
    seed: int = 42,
):
    dataset = IterableDataset.from_generator(
        token_stream,
        gen_kwargs={
            "data_path": str(data_path),
            "sequence_length": sequence_length,
        },
    )

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=shuffle_buffer_size,
    )

    return dataset
