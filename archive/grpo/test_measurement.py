"""Per-row coverage metrics from four graded draws."""


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
