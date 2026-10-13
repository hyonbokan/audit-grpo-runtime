"""The complete corrected-cycle decision refuses omitted or stale evidence."""

import json
from pathlib import Path

import pytest

import cycle_contract
from testing.clock import freeze
from testing.cycles import schedule_cycle

CONDITIONS = [
    "complete_group",
    "known_grades",
    "sampling_support",
    "source_worker_transfer",
    "finite_loss",
    "nonzero_gradient",
    "correction_health",
    "changed_weights",
    "optimizer_progress",
    "checkpoint_saved",
    "fresh_worker_transfer",
    "fresh_weight_version",
    "precision_declared",
    "repeat_drift_retained",
]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_participates_in_complete_decision(condition):
    evidence = dict.fromkeys(CONDITIONS, True)
    evidence[condition] = False
    result = cycle_contract.evaluate_cycle(evidence)
    assert result["approved"] is False
    assert result["failed_conditions"] == [condition]


def test_missing_condition_and_unknown_evidence_cannot_approve_training():
    evidence = dict.fromkeys(CONDITIONS, True)
    del evidence["fresh_worker_transfer"]
    with pytest.raises(ValueError):
        cycle_contract.evaluate_cycle(evidence)
    evidence["fresh_worker_transfer"] = None
    with pytest.raises(ValueError):
        cycle_contract.evaluate_cycle(evidence)


def test_all_explicit_conditions_approve_only_one_cycle():
    result = cycle_contract.evaluate_cycle(dict.fromkeys(CONDITIONS, True))
    assert result == {"approved": True, "failed_conditions": [], "scope": "one_corrected_cycle"}


@pytest.mark.parametrize(
    "name, value",
    [
        ("seconds_limit", 5401),
        ("retry_limit", 1),
        ("judge_operation_cap_usd", 11),
        ("target_step", 3),
        ("native_adapter_precision", "bfloat16"),
        ("optimizer_kind", "adamw_torch"),
        ("correction_max", 100),
        ("max_clip_fraction", 0.99),
        ("min_ess_fraction", 0),
    ],
)
def test_configuration_cannot_expand_reviewed_scope(name, value):
    config = json.loads((Path(__file__).parent / "fixtures/cycle.json").read_text())
    config[name] = value
    with pytest.raises(ValueError):
        cycle_contract.validate_config(config)


def test_reviewed_cycle_allows_time_for_generation_training_and_refresh():
    config = json.loads((Path(__file__).parent / "fixtures/cycle.json").read_text())
    config["seconds_limit"] = 5400
    cycle_contract.validate_config(config)


def test_cli_rejects_missing_checkpoint_without_creating_prepared_receipt(tmp_path, monkeypatch):

    freeze(monkeypatch, cycle_contract, "2026-10-09T00:00:00+00:00")
    config = json.loads((Path(__file__).parent / "fixtures/cycle.json").read_text())
    config["checkpoint"] = str(tmp_path / "absent")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError, match=r"adapter_model\.safetensors"):
        cycle_contract.main(["prepare", "--config", str(path), "--out", str(tmp_path)])
    assert not (tmp_path / "prepared.json").exists()


def cycle_fixture(tmp_path):
    config = json.loads((Path(__file__).parent / "fixtures/cycle.json").read_text())
    repeat = {
        "raw_probability_parity": True,
        "raw_max_abs_logprob_difference": 0.0,
        "processed_max_abs_logprob_difference": 0.0,
    }
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"repeat": repeat}))
    config["prior_analysis"] = str(prior)
    files = {
        "rollout/receipt.json": {
            "transfer": {"cast_transfer_verified": True},
            "weight_version": "step-1:old",
            "native_adapter_precision": "float32",
            "rollout_adapter_precision": "bfloat16",
        },
        "refresh/receipt.json": {
            "transfer": {"cast_transfer_verified": True},
            "weight_version": "step-2:new",
            "native_adapter_precision": "float32",
            "rollout_adapter_precision": "bfloat16",
        },
        "update/receipt.json": {
            "loss": 0.1,
            "gradient_norm": 1.0,
            "correction_clip_fraction": 0.0,
            "correction_ess_fraction": 1.0,
            "adapter_weights_changed": True,
            "optimizer_steps": [2],
            "weight_version": "step-2:new",
            "source_weight_version": "step-1:old",
        },
        "rollout/samples.json": [{"sampling_support_ids": [[1]], "weight_version": "step-1:old"}]
        * 4,
        "grading/rewards.json": [1, 0, 1, -1],
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    checkpoint = tmp_path / "update/checkpoint-2"
    checkpoint.mkdir()
    for name in (
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
    ):
        (checkpoint / name).touch()
    return config, repeat


def test_saved_repeat_evidence_does_not_require_historical_failure(tmp_path):
    config, repeat = cycle_fixture(tmp_path)
    result = cycle_contract.finish_cycle(config, tmp_path)
    assert result["decision"]["approved"] is True
    assert result["prior_repeat"] == repeat
    assert json.loads((tmp_path / "cycle-result.json").read_text())["optimizer_updates"] == 1


def test_fresh_version_cannot_hide_training_from_a_different_source(tmp_path):
    config, _ = cycle_fixture(tmp_path)
    path = tmp_path / "update/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["source_weight_version"] = "step-1:unrelated"
    path.write_text(json.dumps(receipt))
    result = cycle_contract.finish_cycle(config, tmp_path)
    assert result["decision"]["approved"] is False
    assert "fresh_weight_version" in result["decision"]["failed_conditions"]


# --- Schedule-bound cycles


def continuation_config():
    config = json.loads((Path(__file__).parent / "fixtures/cycle.json").read_text())
    config.update(
        schedule_id="continuation-expanded-20261009",
        initial_step=2,
        target_step=3,
        seconds_limit=14400,
        judge_operation_cap_usd=100,
        row_id="new-scope/all-scope",
        dataset_workspace="/results/run/dataset",
    )
    return config


def test_preparation_carries_explicit_dotenv_path_to_training_worker(tmp_path, monkeypatch):
    import hashlib

    from cycle_contract import prepare_cycle

    freeze(monkeypatch, cycle_contract, "2026-10-09T00:00:00+00:00")

    config = continuation_config()
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / name).write_bytes(b"saved")
    (checkpoint / "trainer_state.json").write_text('{"global_step":2}')
    config.update(
        checkpoint=str(checkpoint),
        model_path=str(checkpoint),
        initial_adapter_sha256=hashlib.sha256(b"saved").hexdigest(),
        judge_env_file="/opt/audit-grpo-secrets/judge.env",
    )
    for key in ("source_plan", "request", "prior_analysis"):
        path = tmp_path / f"{key}.json"
        path.write_text("{}")
        config[key] = str(path)
    out = tmp_path / "cycle"
    out.mkdir()
    prepare_cycle(config, out)
    plan = json.loads((out / "judge-plan.json").read_text())
    assert plan["judge_env_file"] == "/opt/audit-grpo-secrets/judge.env"


def test_update_rejects_another_rows_samples_even_if_rewards_and_weights_match():
    from grpo.corrected_update import join_update_group

    config = continuation_config()
    samples = [{"row_id": "other/all-scope", "weight_version": "step-2:digest"}] * 4
    with pytest.raises(ValueError, match="row"):
        join_update_group(config, samples, [1, 0, 1, -1], "step-2:digest")


def test_update_joins_only_four_known_rewards_to_the_selected_row_and_weight():
    from grpo.corrected_update import join_update_group

    config = continuation_config()
    samples = [{"row_id": config["row_id"], "weight_version": "step-2:digest"}] * 4
    group = join_update_group(config, samples, [1, 0, 1, -1], "step-2:digest")
    assert [sample["reward"] for sample in group] == [1, 0, 1, -1]
    for rewards in ([1, 0, None, -1], [1, 0, float("nan"), -1], [1, 0]):
        with pytest.raises(ValueError):
            join_update_group(config, samples, rewards, "step-2:digest")
    with pytest.raises(ValueError):
        join_update_group(config, samples, [1, 0, 1, -1], "step-3:stale")


def test_acceptance_rejects_samples_from_a_row_outside_the_frozen_step(tmp_path):
    from cycle_contract import finish_cycle

    config, _ = cycle_fixture(tmp_path)
    config["row_id"] = "selected/all-scope"
    path = tmp_path / "rollout/samples.json"
    samples = json.loads(path.read_text())
    for sample in samples:
        sample.update(row_id="swapped/all-scope", prompt_ids=[1])
    path.write_text(json.dumps(samples))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"row_id": "selected/all-scope", "prompt_ids": [1]}))
    config["request"] = str(request)
    result = finish_cycle(config, tmp_path)
    assert result["decision"]["approved"] is False
    assert "complete_group" in result["decision"]["failed_conditions"]


def test_rejected_config_names_the_fields_that_differ():
    changed = schedule_cycle("medium-checkpoint4-20261010")
    changed["seed"] = 3408
    with pytest.raises(ValueError, match=r"seed=3408 \(reviewed 3407\)"):
        cycle_contract.validate_config(changed)
    changed = schedule_cycle("medium-checkpoint4-20261010")
    changed["engine"]["seed"] = 3408
    with pytest.raises(ValueError, match=r"engine differs.*seed=3408"):
        cycle_contract.validate_config(changed)


@pytest.mark.parametrize(
    "now,seconds", [("2026-10-10T00:00:00+00:00", 21600), ("2026-10-10T23:00:00+00:00", 7200)]
)
def test_launcher_settings_cap_the_cycle_at_the_absolute_deadline(monkeypatch, now, seconds):
    freeze(monkeypatch, cycle_contract, now)
    config = schedule_cycle("medium-checkpoint-20261010")
    config.update(seconds_limit=21600, absolute_deadline="2026-10-11T01:00:00+00:00")
    monkeypatch.setattr(cycle_contract, "validate_config", lambda _config: None)
    assert cycle_contract.launcher_settings(config) == [
        config["native_image"],
        config["rollout_image"],
        str(seconds),
        str(config["judge_seconds_limit"]),
        "false",
    ]


def test_launcher_settings_report_a_passed_deadline_as_no_time_left(monkeypatch):
    freeze(monkeypatch, cycle_contract, "2026-10-12T00:00:00+00:00")
    config = schedule_cycle("medium-checkpoint-20261010")
    config.update(absolute_deadline="2026-10-11T23:00:00+00:00", skip_uninformative_groups=True)
    settings = cycle_contract.launcher_settings(config)
    assert int(settings[2]) <= 0
    assert settings[4] == "true"
