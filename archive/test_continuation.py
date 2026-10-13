"""Reject stale rows and checkpoints before scheduling the next real update."""

import json
from pathlib import Path

import pytest

from cycle_contract import validate_config


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


def test_continuation_accepts_successive_updates_with_generous_judge_headroom():
    config = continuation_config()
    validate_config(config)
    config.update(initial_step=7, target_step=8)
    config["engine"]["max_model_len"] = 98304
    validate_config(config)


def test_preparation_carries_explicit_dotenv_path_to_training_worker(tmp_path, monkeypatch):
    import hashlib
    from datetime import datetime, timezone
    import cycle_contract
    from cycle_contract import prepare_cycle

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 9, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(cycle_contract, "datetime", FixedClock)

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


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_step": 1},
        {"target_step": 4},
        {"initial_step": 8, "target_step": 9},
        {"judge_operation_cap_usd": 151},
        {"seconds_limit": 21601},
        {"row_id": ""},
        {"dataset_workspace": ""},
    ],
)
def test_continuation_rejects_out_of_bounds_or_unbound_updates(changes):
    config = continuation_config()
    config.update(changes)
    with pytest.raises(ValueError):
        validate_config(config)


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


@pytest.mark.parametrize("tampered", [False, True])
def test_next_step_resumes_only_an_accepted_predecessor_checkpoint(tmp_path, tampered):
    from continuation import prepare_step

    config = continuation_config()
    config.update(
        schedule=[
            {"row_id": "first/all-scope", "request": "first.json", "context": 36864},
            {"row_id": "second/all-scope", "request": "second.json", "context": 53248},
        ]
    )
    previous = tmp_path / "steps/step-3"
    checkpoint = previous / "update/checkpoint-3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"trained-adapter")
    (checkpoint / "trainer_state.json").write_text('{"global_step":3}')
    (previous / "cycle-result.json").write_text(
        json.dumps(
            {
                "decision": {"approved": True},
                "target_step": 3,
            }
        )
    )
    (previous / "update/receipt.json").write_text(
        json.dumps(
            {
                "adapter_sha256": "2249cc966d6609cecce19e333167e3edc54610f01b3777762ab7de131b0363f0",
                "weight_version": "step-3:2249cc966d6609cecce19e333167e3edc54610f01b3777762ab7de131b0363f0",
            }
        )
    )
    if tampered:
        (checkpoint / "adapter_model.safetensors").write_bytes(b"swapped-adapter")
        with pytest.raises(ValueError, match="checkpoint identity"):
            prepare_step(config, tmp_path, 1)
        assert not (tmp_path / "steps/step-4").exists()
        return
    path = prepare_step(config, tmp_path, 1)
    step = json.loads(path.read_text())
    assert (step["initial_step"], step["target_step"]) == (3, 4)
    assert step["row_id"] == "second/all-scope"
    assert step["checkpoint"] == str(checkpoint)
    assert step["initial_adapter_sha256"] == (
        "2249cc966d6609cecce19e333167e3edc54610f01b3777762ab7de131b0363f0"
    )
    assert step["engine"]["max_model_len"] == 53248


def test_failed_predecessor_cannot_schedule_another_update(tmp_path):
    from continuation import prepare_step

    config = continuation_config()
    config["schedule"] = [
        {"row_id": "first", "request": "first.json", "context": 36864},
        {"row_id": "second", "request": "second.json", "context": 36864},
    ]
    previous = tmp_path / "steps/step-3"
    previous.mkdir(parents=True)
    (previous / "cycle-result.json").write_text(
        json.dumps(
            {
                "decision": {"approved": False},
                "target_step": 3,
            }
        )
    )
    with pytest.raises(ValueError):
        prepare_step(config, tmp_path, 1)
    assert not (tmp_path / "steps/step-4/config.json").exists()


def dataset_fixture(tmp_path):
    config = continuation_config()
    config["dataset_workspace"] = str(tmp_path / "dataset")
    private_dir = tmp_path / "dataset/data/private"
    private_dir.mkdir(parents=True)
    row = {
        "row_id": config["row_id"],
        "scorer_metadata": {
            "targets": [
                {"id": "archive:1", "required_context": []},
            ]
        },
    }
    (private_dir / "train.jsonl").write_text(json.dumps(row) + "\n")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "row_id": config["row_id"],
                "prompt_ids": [1, 2],
                "prompt_ids_sha256": "3a316d6d3226f84c1e46e4447fa8d5fd800bff4a1bc6498152523cd4a602b69b",
                "model": {
                    "precision": "bf16",
                    "id": config["model_id"],
                    "revision": config["model_revision"],
                },
                "max_model_len": 36864,
                "full_completion_ceiling": 16384,
            }
        )
    )
    config["schedule"] = [{"row_id": config["row_id"], "request": str(request), "context": 36864}]
    return config, row, private_dir


def test_schedule_rejects_empty_or_context_only_reward_targets(tmp_path):
    from continuation import validate_schedule

    config, row, private_dir = dataset_fixture(tmp_path)
    for targets in ([], [{"id": "archive:1", "required_context": ["documentation"]}]):
        row["scorer_metadata"]["targets"] = targets
        (private_dir / "train.jsonl").write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="eligible"):
            validate_schedule(config)


def test_schedule_rejects_a_prompt_from_another_row(tmp_path):
    from continuation import validate_schedule

    config, _, _ = dataset_fixture(tmp_path)
    path = Path(config["schedule"][0]["request"])
    request = json.loads(path.read_text())
    request["row_id"] = "swapped/all-scope"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="row"):
        validate_schedule(config)


def test_schedule_validates_complete_prompt_and_completion_fit(tmp_path):
    from continuation import validate_schedule

    config, _, _ = dataset_fixture(tmp_path)
    validate_schedule(config)
    path = Path(config["schedule"][0]["request"])
    request = json.loads(path.read_text())
    request["max_model_len"] = 16384
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        validate_schedule(config)


def test_acceptance_rejects_samples_from_a_row_outside_the_frozen_step(tmp_path):
    from cycle_contract import finish_cycle
    from test_cycle_contract import cycle_fixture

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


def test_progress_retains_completed_rows_rewards_and_last_resumable_checkpoint(tmp_path):
    from continuation import record_progress

    config = continuation_config()
    config["run_id"] = "test"
    config["schedule"] = [
        {"row_id": "first", "request": "first.json", "context": 36864},
        {"row_id": "second", "request": "second.json", "context": 36864},
    ]
    step = tmp_path / "steps/step-3"
    for folder in ("rollout", "update", "grading"):
        (step / folder).mkdir(parents=True, exist_ok=True)
    values = {
        "cycle-result.json": {"decision": {"approved": True}, "target_step": 3},
        "rollout/receipt.json": {"completion_tokens": 1234, "generation_seconds": 20.0},
        "update/receipt.json": {
            "loss": -0.01,
            "gradient_norm": 0.7,
            "correction_clip_fraction": 0.0,
            "correction_ess_fraction": 0.8,
            "weight_version": "step-3:new",
            "adapter_sha256": "new",
        },
        "grading/rewards.json": [1, 0, -1, 2],
        "grading/metrics.json": {"row_id": "first", "known_gt_coverage": 0.5},
    }
    for name, value in values.items():
        (step / name).write_text(json.dumps(value))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "calls": [
                    {"id": "r1", "budget_charge_usd": 0.5, "estimated_usd": None, "usage": None},
                    {"id": "unrelated", "budget_charge_usd": 50, "estimated_usd": 10, "usage": {}},
                ]
            }
        )
    )
    call = step / "grading/call-0001"
    call.mkdir()
    (call / "budget-reservation.json").write_text(
        json.dumps(
            {
                "reservation_id": "r1",
                "ledger": str(ledger),
            }
        )
    )
    report = record_progress(config, tmp_path)
    assert report["completed_updates"] == 1
    assert report["sampled_row_ids"] == ["first"]
    assert report["unsampled_row_ids"] == ["second"]
    assert report["last_checkpoint"] == str(step / "update/checkpoint-3")
    assert report["steps"][0]["rewards"] == [1, 0, -1, 2]
    assert report["completion_tokens"] == 1234
    assert report["steps"][0]["judge_budget_charge_usd"] == 0.5
    assert report["steps"][0]["judge_known_usage_estimate_usd"] == 0
    assert report["steps"][0]["judge_unknown_usage_calls"] == 1
