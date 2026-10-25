from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
import random
import tyro

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer

from decider.dataset import build_dataset, split_holdout
from decider.encoder import Encoded, options
from decider.model import DecisionModel
from decider.types import DecisionInput

LOG = logging.getLogger("DECIDER")


@dataclass
class Args:
    seed: int = 42
    model_id: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "2b48083dfa97c3cbf0220cf1a5e6cffe0a511157"
    dataset_id: str = "LocalLLaMA/typed-decisions"
    dataset_name: str | None = "all"
    dataset_split: str = "train"
    dataset_streaming: bool = False
    max_seq_len: int = 2048


def main(args: Args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_id = args.model_id
    revision = args.revision

    model = DecisionModel(
        model_id=model_id,
        revision=revision,
        device=torch.device("mps"),
    )
    tokenizer = model.tokenizer
    processor = model.processor
    decision_codes = model.decision_codes

    # Dataset
    dataset_id = args.dataset_id
    dataset_kwargs = {
        "name": args.dataset_name,
        "streaming": args.dataset_streaming,
        "split": args.dataset_split,
    }
    dataset = load_dataset(dataset_id, **dataset_kwargs)

    


if __name__ == "__main__":
    args = tyro.cli(Args)

    main(args)
