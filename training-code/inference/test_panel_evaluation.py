import json
from pathlib import Path

import pytest

import panel_evaluation as panel
from evaluation import validate_evaluation_config


def config():
    c = json.loads((Path(__file__).parents[1] / "fixtures/evaluation.json").read_text())
    c.update(
        evaluation_kind="grpo_held_out_panel",
        split="calibration",
        dataset_workspace="/dataset",
        request="/requests/row.json",
        judge_operation_cap_usd=150,
        judge_seconds_limit=1800,
    )
    c["cases"] = [{**c["cases"][1], "label": "checkpoint-15", "initial_step": 15}]
    return c


def test_new_panel_accepts_validation_and_later_checkpoints_without_training_masks():
    validate_evaluation_config(config())
    c = config()
    c["split"] = "train"
    with pytest.raises(ValueError):
        validate_evaluation_config(c)
    c = config()
    c["engine"]["return_sampling_mask"] = True
    with pytest.raises(ValueError):
        validate_evaluation_config(c)


def test_per_draw_known_gt_metrics_do_not_use_union_of_four_draws():
    grades = []
    samples = []
    for i in range(4):
        grades.append(
            {
                "reward": 0.0,
                "components": {
                    "format_valid": True,
                    "withheld_reasons": [],
                    "raw_findings": 1,
                    "unambiguous_pairs": [["g0", "c0"]] if i == 0 else [],
                },
                "judge_responses": [
                    {"assessments": [{"gt_id": "g0", "same_defect_operation": True}]}
                ],
            }
        )
        samples.append(
            {"finish_reason": "stop", "final_text": '{"findings":[]}', "reasoning": "trace"}
        )
    metric = panel.project_metrics("row", ["g0", "g1"], grades, samples)
    assert metric["exact_coverage"] == 0.125
    assert metric["partial_coverage"] == 0.5
    assert metric["format_validity"] == 1.0
    grades[0]["reward"] = None
    with pytest.raises(ValueError):
        panel.project_metrics("row", ["g0", "g1"], grades, samples)


def test_unknown_gt_and_unmatched_row_cannot_inflate_coverage():
    grades = [
        {
            "reward": 0.0,
            "components": {
                "format_valid": True,
                "withheld_reasons": [],
                "raw_findings": 1,
                "unambiguous_pairs": [["unknown", "c0"]],
            },
            "judge_responses": [],
        }
    ] * 4
    samples = [{"finish_reason": "stop", "final_text": '{"findings":[]}', "reasoning": "trace"}] * 4
    with pytest.raises(ValueError):
        panel.project_metrics("row", ["g0"], grades, samples)


def test_saved_grades_must_join_the_exact_generated_samples_and_gt_denominator():
    samples = [{"row_id": "r", "final_text": "answer"}] * 4
    metrics = {"row_id": "r", "eligible_gt_ids": ["g"]}
    panel.validate_grading_join(samples, samples, metrics, "r", ["g"])
    with pytest.raises(ValueError):
        panel.validate_grading_join(
            samples, [{"row_id": "r", "final_text": "different"}] * 4, metrics, "r", ["g"]
        )
    with pytest.raises(ValueError):
        panel.validate_grading_join(samples, samples, metrics, "r", ["other"])
