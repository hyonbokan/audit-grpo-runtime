"""Volume-matched recall: the trained arm against the base cut to the same report length."""

from __future__ import annotations

import math
from typing import Any

from benchmark_free_text import mean


def volume_matched_gain(
    base_draws: list[dict[str, Any]], trained_draws: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Compare graded draws only; None when either arm has none."""
    base = [draw for draw in base_draws if draw["status"] == "graded"]
    trained = [draw for draw in trained_draws if draw["status"] == "graded"]
    if not base or not trained:
        return None
    length = mean(draw["findings_count"] for draw in trained)
    trained_recall = mean(draw["exact_matches"] / draw["known_gt_count"] for draw in trained)
    at_length = mean(recall_at_length(draw, length) for draw in base)
    return {
        "trained_mean_findings": length,
        "trained_recall": trained_recall,
        "base_recall_at_trained_length": at_length,
        "base_recall_uncut": mean(draw["exact_matches"] / draw["known_gt_count"] for draw in base),
        "gain": trained_recall - at_length,
        "base_graded_draws": len(base),
        "trained_graded_draws": len(trained),
    }


def recall_at_length(draw: dict[str, Any], length: float) -> float:
    """Exact recall counting only matches among the first `length` findings, interpolated."""
    below = math.floor(length)
    fraction = length - below
    cut = prefix_recall(draw, below)
    if not fraction:
        return cut
    return (1 - fraction) * cut + fraction * prefix_recall(draw, below + 1)


def prefix_recall(draw: dict[str, Any], length: int) -> float:
    matched = sum(
        verdict["is_match"] and 0 <= verdict["index_of_finding_from_junior_auditor"] < length
        for verdict in draw["verdicts"]
    )
    return matched / draw["known_gt_count"]
