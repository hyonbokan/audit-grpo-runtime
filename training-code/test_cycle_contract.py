"""The complete corrected-cycle decision refuses omitted or stale evidence."""

import hashlib
import json

import pytest

import cycle_contract
import schedules
from testing.clock import freeze
from testing.cycles import schedule_cycle

CONDITIONS = cycle_contract.CONDITIONS
# Dropping one of these weakens acceptance, so it takes a deliberate edit here.
REVIEWED_CONDITIONS = {
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
    "saved_adapter_identity",
    "weight_version_chain",
    "precision_declared",
    "repeat_drift_retained",
}


def test_no_reviewed_condition_is_dropped_from_the_decision():
    assert set(CONDITIONS) >= REVIEWED_CONDITIONS


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_participates_in_complete_decision(condition):
    evidence = dict.fromkeys(CONDITIONS, True)
    evidence[condition] = False
    result = cycle_contract.evaluate_cycle(evidence)
    assert result["approved"] is False
    assert result["failed_conditions"] == [condition]


def test_missing_condition_and_unknown_evidence_cannot_approve_training():
    evidence = dict.fromkeys(CONDITIONS, True)
    del evidence["saved_adapter_identity"]
    with pytest.raises(ValueError):
        cycle_contract.evaluate_cycle(evidence)
    evidence["saved_adapter_identity"] = None
    with pytest.raises(ValueError):
        cycle_contract.evaluate_cycle(evidence)


DAY = "medium-day-20261010"
DAY_LIMITS = schedules.schedule(DAY)["training"]


@pytest.mark.parametrize(
    "name, value",
    [
        ("retry_limit", 1),
        ("judge_operation_cap_usd", DAY_LIMITS["judge_operation_cap_usd"] + 1),
        ("native_adapter_precision", "bfloat16"),
        ("optimizer_kind", "adamw_torch"),
        ("correction_max", 100),
        ("max_clip_fraction", 0.99),
        ("min_ess_fraction", 0),
    ],
)
def test_configuration_cannot_expand_reviewed_scope(name, value):
    config = schedule_cycle(DAY)
    cycle_contract.validate_config(config)
    config[name] = value
    with pytest.raises(ValueError):
        cycle_contract.validate_config(config)


@pytest.mark.parametrize("change", [-0.05, 0.05])
def test_engine_memory_allocation_is_the_reviewed_one(change):
    config = schedule_cycle(DAY)
    config["engine"]["gpu_memory_utilization"] += change
    with pytest.raises(ValueError, match="engine"):
        cycle_contract.validate_config(config)


@pytest.mark.parametrize("schedule_id", [None, "continuation-expanded-20261009"])
def test_a_cycle_must_name_a_registered_schedule(schedule_id):
    config = schedule_cycle(DAY)
    config["schedule_id"] = schedule_id
    with pytest.raises(ValueError, match="schedule"):
        cycle_contract.validate_config(config)


def test_cli_rejects_missing_checkpoint_without_creating_prepared_receipt(tmp_path, monkeypatch):
    freeze(monkeypatch, cycle_contract, "2026-10-10T00:00:00+00:00")
    config = schedule_cycle(DAY)
    config["checkpoint"] = str(tmp_path / "absent")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError, match=r"adapter_model\.safetensors"):
        cycle_contract.main(["prepare", "--config", str(path), "--out", str(tmp_path)])
    assert not (tmp_path / "prepared.json").exists()


SAVED_ADAPTER = b"trained adapter"
SAVED_ADAPTER_SHA256 = hashlib.sha256(SAVED_ADAPTER).hexdigest()


def cycle_fixture(tmp_path):
    """A scheduled cycle whose rollout, grades and update all completed and agree."""
    config = schedule_cycle(DAY)
    source, target = config["initial_step"], config["target_step"]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"row_id": config["row_id"], "prompt_ids": [1]}))
    config["request"] = str(request)
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
            "weight_version": f"step-{source}:old",
            "native_adapter_precision": "float32",
            "rollout_adapter_precision": "bfloat16",
        },
        "update/receipt.json": {
            "loss": 0.1,
            "gradient_norm": 1.0,
            "correction_clip_fraction": 0.0,
            "correction_ess_fraction": 1.0,
            "adapter_weights_changed": True,
            "optimizer_steps": [target],
            "adapter_sha256": SAVED_ADAPTER_SHA256,
            "weight_version": f"step-{target}:{SAVED_ADAPTER_SHA256}",
            "source_weight_version": f"step-{source}:old",
        },
        "rollout/samples.json": [
            {
                "row_id": config["row_id"],
                "prompt_ids": [1],
                "sampling_support_ids": [[1]],
                "weight_version": f"step-{source}:old",
            }
        ]
        * 4,
        "grading/rewards.json": [1, 0, 1, -1],
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    checkpoint = tmp_path / f"update/checkpoint-{target}"
    checkpoint.mkdir()
    for name in ("optimizer.pt", "rng_state.pth", "scheduler.pt", "trainer_state.json"):
        (checkpoint / name).touch()
    (checkpoint / "adapter_model.safetensors").write_bytes(SAVED_ADAPTER)
    return config, repeat


def test_saved_repeat_evidence_does_not_require_historical_failure(tmp_path):
    config, repeat = cycle_fixture(tmp_path)
    result = cycle_contract.finish_cycle(config, tmp_path)
    assert result["decision"]["approved"] is True
    assert result["prior_repeat"] == repeat
    assert set(result["evidence"]) == set(CONDITIONS)
    assert json.loads((tmp_path / "cycle-result.json").read_text())["optimizer_updates"] == 1


def test_update_cannot_hide_training_from_a_different_source(tmp_path):
    config, _ = cycle_fixture(tmp_path)
    path = tmp_path / "update/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["source_weight_version"] = "step-0:unrelated"
    path.write_text(json.dumps(receipt))
    result = cycle_contract.finish_cycle(config, tmp_path)
    assert result["decision"]["failed_conditions"] == ["weight_version_chain"]


@pytest.mark.parametrize(
    ("receipt_change", "saved_bytes"),
    [({}, b"a different adapter"), ({"weight_version": "step-4:other"}, SAVED_ADAPTER)],
)
def test_saved_adapter_must_be_the_one_the_update_reported(tmp_path, receipt_change, saved_bytes):
    config, _ = cycle_fixture(tmp_path)
    path = tmp_path / "update/receipt.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **receipt_change}))
    target = config["target_step"]
    (tmp_path / f"update/checkpoint-{target}/adapter_model.safetensors").write_bytes(saved_bytes)
    result = cycle_contract.finish_cycle(config, tmp_path)
    assert result["decision"]["failed_conditions"] == ["saved_adapter_identity"]


# --- Schedule-bound cycles


def continuation_config():
    return dict(
        schedule_cycle(DAY), row_id="new-scope/all-scope", dataset_workspace="/results/run/dataset"
    )


def test_preparation_carries_explicit_dotenv_path_to_training_worker(tmp_path, monkeypatch):
    from cycle_contract import prepare_cycle

    freeze(monkeypatch, cycle_contract, "2026-10-10T00:00:00+00:00")

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
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": config["initial_step"]})
    )
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
    assert cycle_contract.launcher_settings(config) == [
        config["native_image"],
        config["rollout_image"],
        str(seconds),
        str(config["judge_seconds_limit"]),
        "false",
        '"device=0"',
        '"device=0"',
    ]


def test_launcher_settings_report_a_passed_deadline_as_no_time_left(monkeypatch):
    freeze(monkeypatch, cycle_contract, "2026-10-12T00:00:00+00:00")
    config = schedule_cycle("medium-checkpoint-20261010")
    config.update(absolute_deadline="2026-10-11T23:00:00+00:00", skip_uninformative_groups=True)
    settings = cycle_contract.launcher_settings(config)
    assert int(settings[2]) <= 0
    assert settings[4] == "true"


# --- GPU placement

SPLIT = {"gpus": [0, 1], "embedding": "gpu", "max_memory_gib": 44}


@pytest.fixture
def split_schedule(tmp_path, monkeypatch):
    """The medium-day schedule, registered again with a two-GPU update and rollout."""
    registry = json.loads(schedules.REGISTRY.read_text())
    reviewed = registry["schedules"]["medium-day-20261010"]
    training = dict(reviewed["training"], update_placement=SPLIT, tensor_parallel_size=2)
    registry["schedules"]["split"] = dict(reviewed, training=training)
    path = tmp_path / "schedules.json"
    path.write_text(json.dumps(registry))
    monkeypatch.setattr(schedules, "REGISTRY", path)
    return schedule_cycle("split")


def test_a_registered_split_is_validated_and_exposes_both_gpus_to_each_phase(
    split_schedule, monkeypatch
):
    config = split_schedule
    assert config["update_placement"] == SPLIT
    assert config["engine"]["tensor_parallel_size"] == 2
    cycle_contract.validate_config(config)
    freeze(monkeypatch, cycle_contract, "2026-10-10T00:00:00+00:00")
    assert cycle_contract.launcher_settings(config)[5:] == ['"device=0,1"', '"device=0,1"']


@pytest.mark.parametrize(
    "change",
    [
        {"update_placement": {"gpus": [0], "embedding": "offloaded"}},
        {"update_placement": {**SPLIT, "max_memory_gib": 47}},
    ],
)
def test_a_cycle_cannot_change_the_reviewed_update_placement(split_schedule, change):
    with pytest.raises(ValueError, match="update_placement"):
        cycle_contract.validate_config({**split_schedule, **change})


def test_a_cycle_cannot_change_the_reviewed_tensor_parallel_size(split_schedule):
    split_schedule["engine"]["tensor_parallel_size"] = 1
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        cycle_contract.validate_config(split_schedule)
