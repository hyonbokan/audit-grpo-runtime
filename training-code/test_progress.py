import copy
import hashlib
import json

import pytest

import progress


def test_shuffled_passes_cover_every_training_row_before_repetition():
    rows = [f"project-{i}" for i in range(20)]
    order = progress.training_order(rows, epochs=2, seed=3407)
    assert set(order[:20]) == set(rows) and set(order[20:]) == set(rows)
    assert order == progress.training_order(rows, epochs=2, seed=3407)
    assert order[:20] != rows


def panel(value):
    return {
        "macro_exact_coverage": value,
        "row_ids": ["a", "b"],
        "unknown_draws": 0,
        "macro_format_validity": 1.0,
        "macro_partial_coverage": value,
    }


def test_checkpoint_selection_uses_fixed_validation_not_training_or_last_step():
    history = [
        {"checkpoint": "cp2", "panel": panel(0.2)},
        {"checkpoint": "cp3", "panel": panel(0.4)},
        {"checkpoint": "cp4", "panel": panel(0.3)},
    ]
    assert progress.select_checkpoint(history, minimum_improvement=0.01)["checkpoint"] == "cp3"
    changed = copy.deepcopy(history)
    changed[-1]["panel"]["row_ids"] = ["different"]
    with pytest.raises(ValueError):
        progress.select_checkpoint(changed, minimum_improvement=0.01)
    changed = copy.deepcopy(history)
    changed[-1]["panel"]["unknown_draws"] = 1
    with pytest.raises(ValueError):
        progress.select_checkpoint(changed, minimum_improvement=0.01)


def test_no_contrast_is_a_recorded_skip_and_unknown_is_a_failure():
    assert progress.group_decision([-2, -2, -2, -2]) == "skip_no_reward_contrast"
    assert progress.group_decision([-2, -0.1, -2, -2]) == "update"
    for rewards in [[None, 0, 0, 0], [float("nan"), 0, 0, 0], [0, 0, 0]]:
        with pytest.raises(ValueError):
            progress.group_decision(rewards)


def test_validation_averages_projects_instead_of_weighting_large_gt_repositories():
    rows = [
        {
            "row_id": "small",
            "exact_coverage": 1.0,
            "partial_coverage": 1.0,
            "format_validity": 1.0,
            "unknown_draws": 0,
        },
        {
            "row_id": "large",
            "exact_coverage": 0.0,
            "partial_coverage": 0.2,
            "format_validity": 0.5,
            "unknown_draws": 0,
        },
    ]
    result = progress.aggregate_panel(rows)
    assert result["macro_exact_coverage"] == 0.5
    assert result["macro_partial_coverage"] == 0.6
    assert result["macro_format_validity"] == 0.75


def test_changed_corpus_file_fails_before_any_training(tmp_path):
    (tmp_path / "train.jsonl").write_text("original")
    manifest = {"files": {"train.jsonl": hashlib.sha256(b"original").hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    c = {
        "dataset_host": str(tmp_path),
        "dataset_manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    progress.verify_dataset(c)
    (tmp_path / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError):
        progress.verify_dataset(c)
