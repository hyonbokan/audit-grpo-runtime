"""Equal-budget pilot metrics and success/regression decisions."""

import pytest


def test_coverage_unions_gt_and_keeps_unknown_draws_visible():
    from grpo import measurement

    grades = [
        {
            "reward": 1.2,
            "components": {
                "format_valid": True,
                "raw_findings": 1,
                "hits": 1,
                "quality_sum": 1,
                "unambiguous_pairs": [["g0", "c0"]],
                "withheld_reasons": [],
            },
        },
        {
            "reward": 0.95,
            "components": {
                "format_valid": True,
                "raw_findings": 2,
                "hits": 1,
                "quality_sum": 0.2,
                "unambiguous_pairs": [["g0", "c0"]],
                "withheld_reasons": [],
            },
        },
        {
            "reward": None,
            "components": {
                "format_valid": True,
                "raw_findings": 1,
                "hits": None,
                "quality_sum": None,
                "unambiguous_pairs": [["g1", "c1"]],
                "withheld_reasons": ["judge_error"],
            },
        },
        {
            "reward": -2,
            "components": {
                "format_valid": False,
                "raw_findings": 0,
                "hits": None,
                "quality_sum": None,
                "withheld_reasons": [],
            },
        },
    ]
    samples = [{"reasoning": "Sampled reasoning"}] * 4
    result = measurement.group_metrics("row", ["g0", "g1"], grades, samples)
    assert result["known_gt_coverage"] == 0.5
    assert result["caught_gt_ids"] == ["g0"]
    assert result["unknown_draws"] == 1
    assert result["valid_final_trace_rate"] == 0.75
    assert result["mean_findings"] == 1
    assert result["quality_per_hit"] == 0.6


def test_comparison_requires_equal_budgets_and_reports_regression():
    from grpo import measurement

    base = {
        "model": {"revision": "a"},
        "sampling": {"draws_per_row": 4},
        "rows": [
            {
                "row_id": "r",
                "split": "evaluation",
                "metrics": {
                    "known_gt_coverage": 0.5,
                    "valid_final_trace_rate": 1,
                    "quality_per_hit": 0.75,
                    "mean_findings": 2,
                },
            }
        ],
    }
    tuned = {
        **base,
        "rows": [
            {
                "row_id": "r",
                "split": "evaluation",
                "metrics": {
                    "known_gt_coverage": 1,
                    "valid_final_trace_rate": 0.5,
                    "quality_per_hit": 0.8,
                    "mean_findings": 2,
                },
            }
        ],
    }
    thresholds = {
        "min_recall_delta": 0.1,
        "max_validity_drop": 0.1,
        "max_quality_drop": 0.1,
        "max_volume_ratio": 1.5,
    }
    result = measurement.compare(base, tuned, thresholds)
    assert result["accepted"] is False
    assert result["failures"] == ["valid_final_trace_rate"]
    with pytest.raises(ValueError, match="budget"):
        measurement.compare(base, {**tuned, "sampling": {"draws_per_row": 2}}, thresholds)


def test_primary_improvement_is_over_evaluation_rows_including_saturated_baseline():
    from grpo import measurement

    common = {"valid_final_trace_rate": 1, "quality_per_hit": 1, "mean_findings": 2}
    base = {
        "model": {"revision": "a"},
        "sampling": {"draws_per_row": 4},
        "rows": [
            {"row_id": "r0", "split": "evaluation", "metrics": {**common, "known_gt_coverage": 1}},
            {"row_id": "r1", "split": "evaluation", "metrics": {**common, "known_gt_coverage": 0}},
        ],
    }
    tuned = {
        **base,
        "rows": [
            base["rows"][0],
            {
                "row_id": "r1",
                "split": "evaluation",
                "metrics": {**common, "known_gt_coverage": 0.5},
            },
        ],
    }
    result = measurement.compare(
        base,
        tuned,
        {
            "min_recall_delta": 0.1,
            "max_validity_drop": 0.1,
            "max_quality_drop": 0.1,
            "max_volume_ratio": 1.5,
        },
    )
    assert result["accepted"] is True
    assert result["recall_delta"] == 0.25


def test_comparison_rejects_changed_judge_effort_before_scoring():
    from grpo import measurement

    base = {"model": {}, "sampling": {}, "judge": {"reasoning_effort": "high"}, "rows": []}
    tuned = {**base, "judge": {"reasoning_effort": "medium"}}
    with pytest.raises(ValueError, match="judge"):
        measurement.compare(base, tuned, {})


def test_unknown_draws_reject_an_otherwise_improved_comparison():
    from grpo import measurement

    def run(coverage, unknown):
        metrics = {
            "known_gt_coverage": coverage,
            "unknown_draws": unknown,
            "valid_final_trace_rate": 1,
            "quality_per_hit": 1,
            "mean_findings": 2,
        }
        return {
            "model": {},
            "sampling": {},
            "rows": [{"row_id": "r", "split": "evaluation", "metrics": metrics}],
        }

    thresholds = {
        "min_recall_delta": 0.1,
        "max_validity_drop": 0.1,
        "max_quality_drop": 0.1,
        "max_volume_ratio": 1.5,
    }
    assert measurement.compare(run(0, 0), run(1, 0), thresholds)["accepted"] is True
    result = measurement.compare(run(0, 0), run(1, 1), thresholds)
    assert result["failures"] == ["unknown_grades"]
