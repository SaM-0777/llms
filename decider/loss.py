from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F

from decider.encoder import Encoded


class DecisionLoss(NamedTuple):
    total: torch.Tensor
    cross_entropy: torch.Tensor
    brier: torch.Tensor


def decision_loss(
    outputs: list[list[torch.Tensor]],
    chunks: Sequence[Encoded],
    brier_weight: float = 0.0,
):
    cross_entropy_terms = []
    brier_terms = []

    for chunk_outputs, chunk in zip(outputs, chunks, strict=True):
        if chunk.targets is None:
            raise ValueError("Encoded chunk has no training targets")

        for logits, target_values in zip(chunk_outputs, chunk.targets, strict=True):
            target = torch.tensor(
                target_values,
                dtype=torch.float32,
                device=logits.device,
            )

            if logits.ndim != 1:
                raise ValueError("Each question logits tensor must be 1-dimensional")
            if logits.numel() != target.numel():
                raise ValueError("Logit count and target count differ")

            log_probabilities = F.log_softmax(logits.float(), dim=-1)
            probabilities = log_probabilities.exp()
            cross_entropy_terms.append(-(target * log_probabilities).sum())
            brier_terms.append((probabilities - target).square().sum())

    if not cross_entropy_terms:
        raise ValueError("No question losses were produced")

    cross_entropy = torch.stack(cross_entropy_terms).mean()
    brier = torch.stack(brier_terms).mean()
    total = (
        cross_entropy if brier_weight == 0.0 else cross_entropy + brier_weight * brier
    )
    return DecisionLoss(
        total=total,
        cross_entropy=cross_entropy,
        brier=brier,
    )
