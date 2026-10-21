"""Mask-replayed DAPO with explicit, detached token-level inference correction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def support_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    chosen: Sequence[int],
    support: Sequence[Sequence[int]],
    *,
    chunk: int = 32,
) -> torch.Tensor:
    """Project a frozen head only onto recorded support; preserve hidden gradients."""
    import torch

    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("Hidden/head dimensions differ")
    if weight.requires_grad:
        raise ValueError("Sparse support projection requires a frozen LM head")
    if len(support) != len(chosen) or len(chosen) != hidden.shape[0] or not chosen:
        raise ValueError("Completion/support lengths differ")
    if type(chunk) is not int or chunk < 1:
        raise ValueError("Positive projection chunk required")
    for token, ids in zip(chosen, support, strict=True):
        if (
            not ids
            or any(type(value) is not int or not 0 <= value < weight.shape[0] for value in ids)
            or len(set(ids)) != len(ids)
            or token not in ids
        ):
            raise ValueError("Invalid recorded support")
    if hidden.device != weight.device:
        # A layer split must end on the head's GPU; a silent copy would hide a bad device map.
        raise ValueError(f"Final hidden states on {hidden.device}, output head on {weight.device}")
    outputs = []
    for start in range(0, len(chosen), chunk):
        rows = support[start : start + chunk]
        width = max(map(len, rows))
        index = torch.zeros((len(rows), width), dtype=torch.long, device=weight.device)
        valid = torch.zeros_like(index, dtype=torch.bool)
        positions = []
        for row, ids in enumerate(rows):
            index[row, : len(ids)] = torch.tensor(ids, device=index.device)
            valid[row, : len(ids)] = True
            positions.append(ids.index(chosen[start + row]))
        projected = torch.einsum("th,tkh->tk", hidden[start : start + len(rows)], weight[index])
        scores = projected.float().masked_fill(~valid, -float("inf"))
        positions = torch.tensor(positions, device=scores.device)
        outputs.append(scores.log_softmax(-1).gather(1, positions[:, None]).squeeze(1))
    return torch.cat(outputs)


def corrected_loss(
    current: torch.Tensor,
    old: torch.Tensor,
    sampled: torch.Tensor,
    *,
    advantage: float,
    total_tokens: int,
    epsilon: float,
    correction_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the group-token-normalized contribution and detached TIS weights."""
    import torch

    if current.shape != old.shape or current.shape != sampled.shape or current.ndim != 1:
        raise ValueError("Probability shapes differ")
    if not all(torch.isfinite(value).all().item() for value in (current, old, sampled)):
        raise ValueError("Training probabilities must be finite")
    if (
        not math.isfinite(advantage)
        or type(total_tokens) is not int
        or total_tokens < current.numel()
        or not 0 < epsilon < 1
        or not math.isfinite(correction_max)
        or correction_max < 1
    ):
        raise ValueError("Invalid corrected objective configuration")
    old, sampled = old.detach(), sampled.detach()
    correction = (old - sampled).clamp(max=math.log(correction_max)).exp().detach()
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    loss = -(surrogate * correction).sum() / total_tokens
    if not torch.isfinite(loss).item():
        raise ValueError("Corrected loss must be finite")
    return loss, correction
