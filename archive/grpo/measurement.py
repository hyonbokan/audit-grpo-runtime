"""Known-GT coverage of one row's four graded draws."""

from __future__ import annotations


def group_metrics(row_id, gt_ids, grades, samples):
    if len(grades) != 4 or len(samples) != 4:
        raise ValueError("Measurement requires four fixed-budget draws")
    caught = {
        pair[0]
        for grade in grades
        if grade["reward"] is not None
        for pair in grade["components"].get("unambiguous_pairs", [])
    }
    if not caught.issubset(gt_ids):
        raise ValueError("Unknown GT in coverage")
    hits = sum(
        grade["components"].get("hits") or 0 for grade in grades if grade["reward"] is not None
    )
    quality = sum(
        grade["components"].get("quality_sum") or 0
        for grade in grades
        if grade["reward"] is not None
    )
    return {
        "row_id": row_id,
        "draws": 4,
        "eligible_gt_ids": sorted(gt_ids),
        "caught_gt_ids": sorted(caught),
        "known_gt_coverage": len(caught) / len(gt_ids) if gt_ids else None,
        "unknown_draws": sum(bool(grade["components"]["withheld_reasons"]) for grade in grades),
        "valid_final_trace_rate": sum(
            grade["components"]["format_valid"] and bool(sample["reasoning"])
            for grade, sample in zip(grades, samples, strict=True)
        )
        / 4,
        "mean_findings": sum(grade["components"]["raw_findings"] for grade in grades) / 4,
        "quality_per_hit": quality / hits if hits else None,
        "matched_hits": hits,
        "quality_sum": quality,
        "unmatched_findings": sum(
            grade["components"]["raw_findings"] - (grade["components"].get("hits") or 0)
            for grade in grades
            if grade["reward"] is not None and grade["components"]["format_valid"]
        ),
    }
