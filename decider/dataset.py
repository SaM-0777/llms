import json
from typing import Any
from torch.utils.data import Dataset

from decider.encoder import Encoded, Encoder
from decider.types import DecisionInput


def parse(value: Any):
    try:
        return json.loads(value)
    except:
        return value


def to_decision_input(row: dict) -> DecisionInput:
    return DecisionInput(
        state=parse(row["state"]),
        questions=parse(row["questions"]),
        gold=parse(row["gold"]),
    )


class EncodedBatchDataset(Dataset):
    def __init__(self, batches: list[list[Encoded]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> list[Encoded]:
        return self.batches[index]


def collate_encoded_batches(features: list[list[Encoded]]) -> dict[str, Any]:
    if not features:
        raise ValueError("Encoded batch features are empty")

    chunks = [chunk for batch in features for chunk in batch]
    if not chunks:
        raise ValueError("Encoded batch contains no chunks")
    return {"chunks": chunks}


def encode_dataset(
    dataset,
    encoder: Encoder,
    batch_size: int = 16,
    max_length: int = 2048,
):
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")

    encoded_batches: list[list[Encoded]] = []
    pending_rows: list[DecisionInput] = []

    for row_index, dataset_row in enumerate(dataset):
        try:
            pending_rows.append(to_decision_input(dataset_row))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Failed to parse dataset row {row_index}") from exc

        if len(pending_rows) == batch_size:
            encoded_batches.append(
                encoder.prepare_batch(
                    rows=pending_rows,
                    max_length=max_length,
                )
            )
            pending_rows = []

    if pending_rows:
        encoded_batches.append(
            encoder.prepare_batch(
                rows=pending_rows,
                max_length=max_length,
            )
        )

    return encoded_batches
