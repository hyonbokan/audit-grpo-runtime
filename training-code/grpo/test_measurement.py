"""Per-row coverage metrics from a graded group."""

import pytest

from grpo import measurement
from testing.judge import TRUTH_IDS, private_row


def graded(matched_targets, findings):
    return {
        "status": "graded",
        "reward": 0.1,
        "findings_count": findings,
        "exact_matches": len(matched_targets),
        "verdicts": [
            {"target_index": i, "is_match": i in matched_targets, "explanation": "x"}
            for i in range(2)
        ],
    }


def test_coverage_unions_labels_across_draws_and_keeps_unknown_and_truncated_visible():
    grades = [
        graded({0}, 4),
        graded({0}, 2),
        {"status": "truncated", "reward": 0.0},
        {"status": "unknown", "reward": None, "error": "ValueError: x"},
    ]
    result = measurement.group_metrics(private_row(), grades)
    assert result["row_id"] == "audit/group-123"
    assert result["eligible_gt_ids"] == TRUTH_IDS
    assert result["caught_gt_ids"] == ["archive:1"]
    assert result["known_gt_coverage"] == 0.5
    assert (result["unknown_draws"], result["truncated_draws"]) == (1, 1)
    assert result["mean_findings"] == 3.0
    assert result["mean_exact_matches"] == 1.0


def test_a_group_with_no_graded_draw_has_no_findings_mean():
    grades = [{"status": "truncated", "reward": 0.0}] * 4
    assert measurement.group_metrics(private_row(), grades)["mean_findings"] is None


def test_a_group_of_other_than_four_draws_is_refused():
    with pytest.raises(ValueError, match="four"):
        measurement.group_metrics(private_row(), [graded(set(), 1)] * 3)
