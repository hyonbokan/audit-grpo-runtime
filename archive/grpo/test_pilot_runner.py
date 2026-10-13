"""Baseline provenance and genuine-step gates for background phase progression."""

import json

import pytest


def test_freeze_baseline_uses_target_denominator_and_medium_request_provenance(tmp_path):
    from grpo.pilot_runner import freeze_baseline

    plan_path, baseline, workspace = prepared_baseline(tmp_path)
    freeze_baseline(plan_path, baseline, workspace)
    plan = json.loads(plan_path.read_text())
    assert plan["thresholds"]["min_recall_delta"] == pytest.approx(1 / 3)
    assert plan["thresholds"]["max_validity_drop"] == 0.1
    assert plan["baseline"]["path"] == str(baseline / "measurement.json")
    assert json.loads((baseline / "measurement.json").read_text())["judge"] == plan["judge"]


@pytest.mark.parametrize("bad_effort", ("high", "low"))
def test_freeze_baseline_rejects_inconsistent_judge_calls(tmp_path, bad_effort):
    from grpo.pilot_runner import freeze_baseline

    plan_path, baseline, workspace = prepared_baseline(tmp_path)
    request = baseline / "group-0001/grading/call-0001/request.json"
    data = json.loads(request.read_text())
    data["reasoning_effort"] = bad_effort
    request.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="judge"):
        freeze_baseline(plan_path, baseline, workspace)
    assert json.loads(plan_path.read_text())["baseline"] is None


def test_step_gate_requires_weight_change_contrast_and_complete_checkpoint(tmp_path):
    from grpo.pilot_runner import validate_step

    (tmp_path / "step_metrics.json").write_text(
        json.dumps({"global_step": 1, "adapter_weights_changed": True})
    )
    checkpoint = tmp_path / "checkpoints/checkpoint-1"
    checkpoint.mkdir(parents=True)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        (checkpoint / name).write_text("saved")
    group = tmp_path / "group-0001/grading"
    group.mkdir(parents=True)
    (group / "rewards.json").write_text("[1.2, 0.95, 0, -2]")
    assert validate_step(tmp_path) == checkpoint
    (group / "rewards.json").write_text("[0, 0, 0, 0]")
    with pytest.raises(ValueError, match="contrast"):
        validate_step(tmp_path)
    (group / "rewards.json").write_text("[1.2, 0.95, 0, -2]")
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        validate_step(tmp_path)


def test_exited_baseline_without_measurement_stops_before_gpu_phase(tmp_path):
    from grpo.pilot_runner import main

    plan_path, baseline, workspace = prepared_baseline(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["access"]["local_experiment_stop"] = {"deadline_at_utc": "2100-01-01T00:00:00Z"}
    plan_path.write_text(json.dumps(plan))
    (baseline / "measurement.json").unlink()
    out, step = tmp_path / "runner", tmp_path / "step"
    with pytest.raises(FileNotFoundError):
        main(
            [
                "--workspace",
                str(workspace),
                "--plan",
                str(plan_path),
                "--baseline",
                str(baseline),
                "--baseline-pid",
                "999999999",
                "--out",
                str(out),
                "--step-out",
                str(step),
                "--gpu-python",
                "/nonexistent/gpu-python",
                "--judge-python",
                "/nonexistent/judge-python",
            ]
        )
    assert json.loads((out / "status.json").read_text())["phase"] == "stopped_on_failure"
    assert not step.exists()
    assert json.loads(plan_path.read_text())["baseline"] is None


@pytest.mark.parametrize(("global_step", "changed"), ((8, True), (1, True), (8, False)))
def test_training_first_followup_resumes_checkpoint_and_requires_eight_real_steps(
    tmp_path, monkeypatch, global_step, changed
):
    import subprocess

    from grpo import pilot_runner
    from grpo.test_trainer import _revised_budget_plan

    plan_path, baseline, workspace = prepared_baseline(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["access"] = _revised_budget_plan()["access"]
    plan["access"]["local_experiment_stop"] = {"deadline_at_utc": "2100-01-01T00:00:00Z"}
    plan["training_first"] = {"authorized": True, "source": "User asked to fine-tune now"}
    plan_path.write_text(json.dumps(plan))
    (baseline / "measurement.json").unlink()
    step = tmp_path / "step"
    checkpoint = step / "checkpoints/checkpoint-1"
    checkpoint.mkdir(parents=True)
    (step / "step_metrics.json").write_text(
        json.dumps({"global_step": 1, "adapter_weights_changed": True})
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        (checkpoint / name).write_text("saved")
    grading = step / "group-0001/grading"
    grading.mkdir(parents=True)
    (grading / "rewards.json").write_text("[1.2, 0.95, 0, -2]")
    pilot, out = tmp_path / "pilot", tmp_path / "followup"

    def gpu_process(command, **_options):
        assert command[command.index("--resume") + 1] == str(checkpoint)
        assert "pilot" in command and "baseline" not in command
        pilot.mkdir()
        (pilot / "step_metrics.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "adapter_weights_changed": changed,
                    "resume_from": str(checkpoint),
                }
            )
        )
        adapter = pilot / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_text("saved adapter")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pilot_runner.subprocess, "run", gpu_process)
    arguments = [
        "--workspace",
        str(workspace),
        "--plan",
        str(plan_path),
        "--step-pid",
        "999999999",
        "--out",
        str(out),
        "--step-out",
        str(step),
        "--pilot-out",
        str(pilot),
        "--gpu-python",
        "/gpu/python",
        "--judge-python",
        "/judge/python",
    ]
    if global_step == 8 and changed:
        assert pilot_runner.main(arguments) == 0
        assert json.loads((out / "status.json").read_text())["phase"] == "pilot_verified"
    else:
        with pytest.raises(ValueError, match="pilot"):
            pilot_runner.main(arguments)
        assert json.loads((out / "status.json").read_text())["phase"] == "stopped_on_failure"
    assert json.loads(plan_path.read_text())["baseline"] is None


def prepared_baseline(root):
    import hashlib

    from grpo import trainer

    workspace = root / "workspace"
    (workspace / "configs").mkdir(parents=True)
    contract = workspace / "configs/reward_contract.json"
    contract.write_text(
        json.dumps({"judge": {"model": "gpt-6.1-sol", "reasoning_effort": "medium"}})
    )
    plan = trainer.default_plan()
    plan["baseline_row_ids"] = ["eval"]
    plan["judge"] = {
        "model": "gpt-6.1-sol",
        "reasoning_effort": "medium",
        "reward_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    }
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    baseline = root / "baseline"
    call = baseline / "group-0001/grading/call-0001"
    call.mkdir(parents=True)
    (call / "request.json").write_text(
        json.dumps({"model": "gpt-6.1-sol", "reasoning_effort": "medium"})
    )
    (baseline / "measurement.json").write_text(
        json.dumps(
            {
                "model": plan["model"],
                "sampling": trainer.sampling(plan),
                "optimizer_steps": 0,
                "rows": [
                    {
                        "row_id": "eval",
                        "split": "evaluation",
                        "metrics": {
                            "draws": 4,
                            "eligible_gt_ids": ["a", "b", "c"],
                            "unknown_draws": 0,
                        },
                    }
                ],
            }
        )
    )
    return plan_path, baseline, workspace
