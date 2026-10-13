import copy
import json
from pathlib import Path
import pytest
import cycle_contract
import experiment


def test_reviewed_full_training_allows_steps_beyond_the_six_row_pilot():
    config = json.loads((Path(__file__).parent / "fixtures/cycle.json").read_text())
    config.update(
        schedule_id="full-training-20261009",
        initial_step=15,
        target_step=16,
        row_id="new/all-scope",
        dataset_workspace="/dataset",
        judge_operation_cap_usd=150,
    )
    cycle_contract.validate_config(config)
    config["target_step"] = 200
    with pytest.raises(ValueError):
        cycle_contract.validate_config(config)


def test_shuffled_passes_cover_every_training_row_before_repetition():
    assert hasattr(experiment, "training_order"), "Expanded pass schedule is missing"
    rows = [f"project-{i}" for i in range(20)]
    order = experiment.training_order(rows, epochs=2, seed=3407)
    assert set(order[:20]) == set(rows) and set(order[20:]) == set(rows)
    assert order == experiment.training_order(rows, epochs=2, seed=3407)
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
    assert hasattr(experiment, "select_checkpoint"), "Validation selection is missing"
    history = [
        {"checkpoint": "cp2", "panel": panel(0.2)},
        {"checkpoint": "cp3", "panel": panel(0.4)},
        {"checkpoint": "cp4", "panel": panel(0.3)},
    ]
    assert experiment.select_checkpoint(history, minimum_improvement=0.01)["checkpoint"] == "cp3"
    changed = copy.deepcopy(history)
    changed[-1]["panel"]["row_ids"] = ["different"]
    with pytest.raises(ValueError):
        experiment.select_checkpoint(changed, minimum_improvement=0.01)
    changed = copy.deepcopy(history)
    changed[-1]["panel"]["unknown_draws"] = 1
    with pytest.raises(ValueError):
        experiment.select_checkpoint(changed, minimum_improvement=0.01)


def test_no_contrast_is_a_recorded_skip_and_unknown_is_a_failure():
    assert hasattr(experiment, "group_decision"), "Uninformative group handling is missing"
    assert experiment.group_decision([-2, -2, -2, -2]) == "skip_no_reward_contrast"
    assert experiment.group_decision([-2, -0.1, -2, -2]) == "update"
    for rewards in [[None, 0, 0, 0], [float("nan"), 0, 0, 0], [0, 0, 0]]:
        with pytest.raises(ValueError):
            experiment.group_decision(rewards)


def test_validation_averages_projects_instead_of_weighting_large_gt_repositories():
    assert hasattr(experiment, "aggregate_panel"), "Macro validation aggregation is missing"
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
    result = experiment.aggregate_panel(rows)
    assert result["macro_exact_coverage"] == 0.5
    assert result["macro_partial_coverage"] == 0.6
    assert result["macro_format_validity"] == 0.75


def test_changed_corpus_file_fails_before_any_training():
    import tempfile
    import hashlib

    assert hasattr(experiment, "verify_dataset"), "Frozen corpus file verification is missing"
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        (root / "train.jsonl").write_text("original")
        manifest = {"files": {"train.jsonl": hashlib.sha256(b"original").hexdigest()}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        c = {
            "dataset_host": name,
            "dataset_manifest_sha256": hashlib.sha256(
                (root / "manifest.json").read_bytes()
            ).hexdigest(),
        }
        experiment.verify_dataset(c)
        (root / "train.jsonl").write_text("changed")
        with pytest.raises(ValueError):
            experiment.verify_dataset(c)


def test_experiment_state_records_a_skip_and_locks_best_before_test(tmp_path):
    import hashlib

    root = tmp_path / "runtime"
    out = root / "results" / "full"
    out.mkdir(parents=True)
    data = root / "dataset"
    data.mkdir()
    (data / "row").write_text("frozen")
    manifest = {
        "files": {"row": hashlib.sha256(b"frozen").hexdigest()},
        "roles": {"train": ["t"], "calibration": ["v1", "v2"], "evaluation": ["e1", "e2"]},
        "requests": {"t": {"container_path": "/requests/t", "context": 36864}},
    }
    experiment.save(data / "manifest.json", manifest)
    c = {
        "runtime_host": str(root),
        "dataset_host": str(data),
        "dataset_container": "/dataset",
        "dataset_manifest_sha256": hashlib.sha256(
            (data / "manifest.json").read_bytes()
        ).hexdigest(),
        "training_row_ids": ["t"],
        "validation_row_ids": ["v1", "v2"],
        "test_row_ids": ["e1", "e2"],
        "epochs": 1,
        "seed": 3407,
        "retry_limit": 0,
        "validation_interval": 2,
        "patience": 2,
        "minimum_improvement": 0.01,
        "initial_checkpoint": "cp2",
        "initial_adapter_sha256": "hash2",
        "training_cutoff": "2099-01-01T00:00:00+00:00",
        "minimum_cycle_seconds": 1,
        "cycle": {"engine": {}, "seconds_limit": 5400},
    }
    (out / "SOURCE_SHA256SUMS").write_text("source hashes")
    experiment.prepare(c, out)
    with pytest.raises(ValueError):
        experiment.make_panel_configs(c, out, "test", "selected")
    visit = Path(experiment.next_training(c, out))
    experiment.save(visit / "grading/rewards.json", [-2] * 4)
    experiment.save(visit / "grading/metrics.json", {"row_id": "t"})
    experiment.save(visit / "rollout/receipt.json", {"completion_tokens": 100})
    experiment.save(visit / "group-decision.json", {"decision": "skip_no_reward_contrast"})
    (visit / "launcher-exit.txt").write_text("0")
    experiment.record_training(c, out)
    state = experiment.read(out / "state.json")
    assert state["current_step"] == 2 and len(state["visits"]) == 1
    state.update(
        current_checkpoint="cp3",
        current_step=3,
        current_adapter_sha256="hash3",
        validation_history=[
            {"checkpoint": "cp2", "step": 2, "adapter_sha256": "hash2", "panel": panel(0.4)},
            {"checkpoint": "cp3", "step": 3, "adapter_sha256": "hash3", "panel": panel(0.2)},
        ],
    )
    experiment.save(out / "state.json", state)
    lock = experiment.lock_selection(c, out)
    assert lock["checkpoint"] == "cp2"
    assert experiment.read(out / "state.json")["training_complete"] is True
    with pytest.raises(FileExistsError):
        experiment.lock_selection(c, out)


def test_final_report_uses_locked_selection_and_paired_per_project_test_deltas():
    assert hasattr(experiment, "test_comparison"), "Final paired test report is missing"
    baseline = experiment.aggregate_panel(
        [
            {
                "row_id": "a",
                "exact_coverage": 0.25,
                "partial_coverage": 0.25,
                "format_validity": 1.0,
                "unknown_draws": 0,
            },
            {
                "row_id": "b",
                "exact_coverage": 0.5,
                "partial_coverage": 0.5,
                "format_validity": 1.0,
                "unknown_draws": 0,
            },
        ]
    )
    selected = experiment.aggregate_panel(
        [
            {
                "row_id": "a",
                "exact_coverage": 0.5,
                "partial_coverage": 0.5,
                "format_validity": 1.0,
                "unknown_draws": 0,
            },
            {
                "row_id": "b",
                "exact_coverage": 0.5,
                "partial_coverage": 0.5,
                "format_validity": 1.0,
                "unknown_draws": 0,
            },
        ]
    )
    result = experiment.test_comparison(baseline, selected)
    assert result["macro_exact_delta"] == 0.125
    assert result["per_project_exact_delta"] == {"a": 0.25, "b": 0.0}
    assert result["quality_gain_established"] is False
    changed = copy.deepcopy(selected)
    changed["row_ids"] = ["a", "other"]
    with pytest.raises(ValueError):
        experiment.test_comparison(baseline, changed)
