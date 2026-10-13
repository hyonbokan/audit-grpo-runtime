"""Native filtering shared by both backends; tensor imports stay lazy for preparation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class SamplingRecorder:
    """Retain chosen-token probabilities and sparse support, never vocabulary history."""

    def __init__(self, *, top_k: int = 20, top_p: float = 0.95) -> None:
        if type(top_k) is not int or top_k < 1 or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("Invalid native sampling settings")
        self.top_k = top_k
        self.top_p = top_p
        self.previous = None
        self.previous_raw = None
        self.previous_support = None
        self.logprobs: list[float] = []
        self.raw_logprobs: list[float] = []
        self.sampling_support_ids: list[list[int]] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        import torch

        if self.previous is not None:
            self.finish(input_ids[0, -1].item())
        filtered = filter_logits(
            scores,
            torch.tensor([min(self.top_k, scores.shape[-1])], device=scores.device),
            torch.tensor([self.top_p], device=scores.device),
        )
        self.previous_raw = scores.float().log_softmax(-1)[0]
        self.previous = filtered.log_softmax(-1)[0]
        self.previous_support = torch.where(torch.isfinite(filtered[0]))[0].tolist()
        return filtered

    def finish(self, token: int) -> None:
        if self.previous is None or token not in self.previous_support:
            raise ValueError("Chosen token is outside the recorded sampling support")
        self.logprobs.append(self.previous[token].item())
        self.raw_logprobs.append(self.previous_raw[token].item())
        self.sampling_support_ids.append(self.previous_support)
        self.previous = self.previous_raw = self.previous_support = None


def filter_logits(
    logits: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
) -> torch.Tensor:
    """Preserve native threshold ties and Torch ordering at the top-p boundary."""
    if logits.ndim != 2:
        raise ValueError("Sampling logits must be a batch by vocabulary matrix")
    scores = logits.float().clone()
    if top_k is not None:
        ranks = top_k.long().clamp(min=1, max=scores.shape[-1])
        maximum = int(ranks.max().item())
        cutoff = scores.topk(maximum, dim=-1).values.gather(1, (ranks - 1)[:, None])
        scores.masked_fill_(scores < cutoff, -float("inf"))
    if top_p is not None:
        ordered, indices = scores.sort(dim=-1, descending=False)
        remove = ordered.softmax(-1).cumsum(-1) <= (1 - top_p[:, None])
        remove[:, -1] = False
        scores.masked_fill_(remove.scatter(1, indices, remove), -float("inf"))
    return scores
