"""Known-label coverage of one row's four graded draws."""

from __future__ import annotations

from typing import Any

GROUP_DRAWS = 4


def group_metrics(row: dict[str, Any], grades: list[dict[str, Any]]) -> dict[str, Any]:
    if len(grades) != GROUP_DRAWS:
        raise ValueError("Measurement requires four fixed-budget draws")
    graded = [grade for grade in grades if grade["status"] == "graded"]
    caught = {
        verdict["target_index"]
        for grade in graded
        for verdict in grade["verdicts"]
        if verdict["is_match"]
    }
    count = len(row["truth_ids"])
    return {
        "row_id": row["row_id"],
        "draws": len(grades),
        "eligible_gt_ids": row["truth_ids"],
        "caught_gt_ids": [row["truth_ids"][index] for index in sorted(caught)],
        "known_gt_coverage": len(caught) / count,
        "unknown_draws": sum(grade["status"] == "unknown" for grade in grades),
        "truncated_draws": sum(grade["status"] == "truncated" for grade in grades),
        "mean_findings": mean(grade["findings_count"] for grade in graded),
        "mean_exact_matches": mean(grade["exact_matches"] for grade in graded),
    }


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None
