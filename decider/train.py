from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
import random
import tyro

import torch
from datasets import get_dataset_split_names, load_dataset
from transformers import AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer

from decider.dataset import build_dataset, split_holdout
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

    row = next(iter(dataset))
    print("keys ", row.keys())
    questions = json.loads(row["questions"])
    batch_input = DecisionInput(questions=questions, state=row["state"])
    chunks = model.encoder.prepare_batch(
        row=batch_input,
    )
    outputs = model(chunks)
    assert len(outputs) == len(chunks)
    for chunk_outputs, chunk in zip(outputs, chunks):
        assert len(chunk_outputs) == len(chunk.keys)

        for question_logits, count in zip(
            chunk_outputs,
            chunk.counts,
        ):
            assert isinstance(question_logits, torch.Tensor)
            assert question_logits.shape == (count,)

    model.zero_grad(set_to_none=True)

    probe_loss = torch.zeros(
        (),
        device=model.device,
        dtype=torch.float32,
    )

    for chunk_outputs in outputs:
        for question_logits in chunk_outputs:
            probe_loss = probe_loss + question_logits.float().sum()

    assert probe_loss.requires_grad

    probe_loss.backward()

    assert model.readout.weight.requires_grad
    assert model.readout.weight.grad is not None
    assert torch.isfinite(model.readout.weight.grad).all()
    assert model.readout.weight.grad.abs().sum().item() > 0

    for name, parameter in model.backbone.named_parameters():
        assert parameter.requires_grad is False
        assert parameter.grad is None

    # input_ids = [enc.input_ids for enc in batch]
    ## print(batch)
    ## input_ids = batch.inputs["input_ids"][0]
    # text = tokenizer.decode(input_ids)
    # print("\n\n".join(text))

    # items, total, skipped = build_dataset(
    #    dataset, tokenizer, max_length=args.max_seq_len
    # )
    # train_items, eval_items = split_holdout(items, seed=3407)
    # print(f"{len(train_items)} training rows, {len(eval_items)} held out")


if __name__ == "__main__":
    args = tyro.cli(Args)

    main(args)
