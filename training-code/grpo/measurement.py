"""Known-GT coverage and explicit equal-budget base/tuned comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grpo.samples import write_new


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


def compare(base, tuned, thresholds):
    if base.get("judge") != tuned.get("judge"):
        raise ValueError("Baseline and tuned judge contracts differ")
    if base["model"] != tuned["model"] or base["sampling"] != tuned["sampling"]:
        raise ValueError("Checkpoint/precision or generation budget differs")
    left = {row["row_id"]: row for row in base["rows"] if row["split"] == "evaluation"}
    right = {row["row_id"]: row for row in tuned["rows"] if row["split"] == "evaluation"}
    if not left or set(left) != set(right):
        raise ValueError("Evaluation row budget differs")
    failures, changes = [], {}
    recall_weighted_delta = denominator = 0
    for identity in left:
        a, b = left[identity]["metrics"], right[identity]["metrics"]
        if a.get("eligible_gt_ids") != b.get("eligible_gt_ids"):
            raise ValueError("Known-GT denominator differs")
        row = {}
        for metric, threshold, sign in (
            ("valid_final_trace_rate", "max_validity_drop", -1),
            ("quality_per_hit", "max_quality_drop", -1),
        ):
            delta = (
                b[metric] - a[metric] if a[metric] is not None and b[metric] is not None else None
            )
            row[metric] = delta
            if delta is None or delta < sign * thresholds[threshold]:
                failures.append(metric)
        ratio = (
            b["mean_findings"] / a["mean_findings"]
            if a["mean_findings"]
            else (1 if not b["mean_findings"] else None)
        )
        row["volume_ratio"] = ratio
        if ratio is None or ratio > thresholds["max_volume_ratio"]:
            failures.append("finding_volume")
        if a.get("unknown_draws", 0) or b.get("unknown_draws", 0):
            failures.append("unknown_grades")
        changes[identity] = row
        a_recall, b_recall = a["known_gt_coverage"], b["known_gt_coverage"]
        if a_recall is None or b_recall is None:
            failures.append("known_gt_coverage")
        else:
            weight = len(a.get("eligible_gt_ids", [])) or 1
            recall_weighted_delta += (b_recall - a_recall) * weight
            denominator += weight
            row["known_gt_coverage"] = b_recall - a_recall
    recall_delta = recall_weighted_delta / denominator if denominator else None
    if recall_delta is None or recall_delta < thresholds["min_recall_delta"]:
        failures.append("known_gt_coverage")
    return {
        "accepted": not failures,
        "failures": sorted(set(failures)),
        "changes": changes,
        "thresholds": thresholds,
        "production_promotion": False,
        "recall_delta": recall_delta,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(
        json.loads(args.base.read_text()),
        json.loads(args.tuned.read_text()),
        json.loads(args.plan.read_text())["thresholds"],
    )
    write_new(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
