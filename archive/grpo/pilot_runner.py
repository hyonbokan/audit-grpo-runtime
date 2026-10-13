"""Progress from baseline to a step, or resume a verified training-first step to eight."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

from grpo.samples import write_new
from grpo.trainer import digest, judging_contract, sampling, validate_live


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-pid", type=int)
    parser.add_argument("--step-pid", type=int)
    parser.add_argument("--pilot-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step-out", type=Path, required=True)
    parser.add_argument("--gpu-python", type=Path, required=True)
    parser.add_argument("--judge-python", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.step_pid is None and (args.baseline is None or args.baseline_pid is None):
        parser.error("Select an active baseline or step process")
    if args.step_pid is not None and args.pilot_out is None:
        parser.error("Step followup requires --pilot-out")
    args.out.mkdir(parents=True, exist_ok=False)
    state = {
        "phase": "waiting_for_caviar_step" if args.step_pid is not None else "waiting_for_baseline",
        "baseline": str(args.baseline) if args.baseline else None,
        "step": str(args.step_out),
    }
    record_status(args.out, state)
    plan = json.loads(args.plan.read_text())
    deadline = datetime.datetime.fromisoformat(
        plan["access"]["local_experiment_stop"]["deadline_at_utc"]
    )
    try:
        if args.step_pid is not None:
            resume_pilot(args, deadline, state)
            return 0
        while baseline_running(args.baseline_pid, args.baseline):
            seconds_left(deadline)
            time.sleep(10)
        seconds_left(deadline)
        freeze_baseline(args.plan, args.baseline, args.workspace)
        plan = json.loads(args.plan.read_text())
        validate_live(plan, phase="step")
        if plan["fit_train_row_ids"] != ["2022-12-caviar/all-scope"]:
            raise ValueError("Next phase requires the selected complete Caviar row")
        state.update(phase="caviar_step_running", thresholds=plan["thresholds"])
        record_status(args.out, state)
        command = [
            str(args.gpu_python),
            "-u",
            "-m",
            "grpo.trainer",
            "step",
            "--run-live",
            "--workspace",
            str(args.workspace),
            "--plan",
            str(args.plan),
            "--judge-python",
            str(args.judge_python),
            "--out",
            str(args.step_out),
        ]
        with (args.out / "step.log").open("x") as log:
            subprocess.run(
                command,
                cwd=args.workspace / "code/backend",
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=seconds_left(deadline),
                check=True,
            )
        checkpoint = validate_step(args.step_out)
        state.update(
            phase="caviar_step_verified",
            checkpoint=str(checkpoint),
            next="resume eight total steps",
        )
        record_status(args.out, state)
    except Exception as exc:
        state.update(phase="stopped_on_failure", error_type=type(exc).__name__, error=str(exc))
        record_status(args.out, state)
        raise
    return 0


def resume_pilot(args, deadline, state):
    while training_running(args.step_pid, args.step_out):
        seconds_left(deadline)
        time.sleep(10)
    seconds_left(deadline)
    checkpoint = validate_step(args.step_out)
    plan = json.loads(args.plan.read_text())
    validate_live(plan, phase="pilot")
    state.update(phase="pilot_running", resume_from=str(checkpoint))
    record_status(args.out, state)
    command = [
        str(args.gpu_python),
        "-u",
        "-m",
        "grpo.trainer",
        "pilot",
        "--run-live",
        "--workspace",
        str(args.workspace),
        "--plan",
        str(args.plan),
        "--judge-python",
        str(args.judge_python),
        "--resume",
        str(checkpoint),
        "--out",
        str(args.pilot_out),
    ]
    with (args.out / "pilot.log").open("x") as log:
        subprocess.run(
            command,
            cwd=args.workspace / "code/backend",
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=seconds_left(deadline),
            check=True,
        )
    metrics = json.loads((args.pilot_out / "step_metrics.json").read_text())
    if (
        metrics["global_step"] != plan["training"]["max_steps"]
        or metrics["adapter_weights_changed"] is not True
        or metrics["resume_from"] != str(checkpoint)
        or not (args.pilot_out / "adapter/adapter_model.safetensors").is_file()
    ):
        raise ValueError("Resumed pilot did not complete verified optimizer progression")
    state.update(
        phase="pilot_verified",
        global_step=metrics["global_step"],
        adapter=str(args.pilot_out / "adapter"),
    )
    record_status(args.out, state)


def freeze_baseline(plan_path, baseline, workspace):
    plan = json.loads(plan_path.read_text())
    expected = judging_contract(workspace)
    if plan.get("judge") != expected:
        raise ValueError("Live judge contract changed")
    measurement = baseline / "measurement.json"
    data = json.loads(measurement.read_text())
    if plan.get("baseline"):
        if plan["baseline"] != {
            "path": str(measurement),
            "sha256": digest(measurement.read_bytes()),
        }:
            raise ValueError("Frozen baseline changed")
        return
    if (
        data["model"] != plan["model"]
        or data["sampling"] != sampling(plan)
        or data["optimizer_steps"] != 0
    ):
        raise ValueError("Baseline model, sampling or optimizer state differs")
    if data.get("judge") not in (None, expected):
        raise ValueError("Baseline judge differs")
    if {row["row_id"] for row in data["rows"]} != set(plan["baseline_row_ids"]):
        raise ValueError("Baseline row set differs")
    if any(row["metrics"]["unknown_draws"] or row["metrics"]["draws"] != 4 for row in data["rows"]):
        raise ValueError("Baseline contains unknown or incomplete groups")
    provenance = []
    for request in sorted(baseline.glob("group-*/grading/call-*/request.json")):
        saved = json.loads(request.read_text())
        if any(saved[key] != expected[key] for key in ("model", "reasoning_effort")):
            raise ValueError("Baseline judge request differs")
        provenance.append(
            {"path": str(request.relative_to(baseline)), "sha256": digest(request.read_bytes())}
        )
    denominator = sum(
        len(row["metrics"]["eligible_gt_ids"])
        for row in data["rows"]
        if row["split"] == "evaluation"
    )
    if not denominator:
        raise ValueError("Baseline has no evaluation targets")
    write_new(baseline / "measurement.before-judge-provenance.json", data)
    data.update(judge=expected, judge_request_manifest=provenance)
    atomic_json(measurement, data)
    plan["baseline"] = {"path": str(measurement), "sha256": digest(measurement.read_bytes())}
    plan["thresholds"] = {
        "min_recall_delta": 1 / denominator,
        "max_validity_drop": 0.1,
        "max_quality_drop": 0.1,
        "max_volume_ratio": 1.25,
    }
    plan["threshold_basis"] = {
        "evaluation_targets": denominator,
        "minimum_improvement": "one additional known GT defect",
        "guardrails": "at most 0.1 validity/quality drop and 25% finding-volume growth",
        "frozen_before_optimizer": True,
    }
    atomic_json(plan_path, plan)


def validate_step(out):
    metrics = json.loads((out / "step_metrics.json").read_text())
    if metrics["global_step"] != 1 or metrics["adapter_weights_changed"] is not True:
        raise ValueError("First step did not change adapter weights")
    checkpoint = out / "checkpoints/checkpoint-1"
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Incomplete resumable checkpoint: {name}")
    groups = [json.loads(path.read_text()) for path in out.glob("group-*/grading/rewards.json")]
    if not groups or not any(max(values) > min(values) for values in groups):
        raise ValueError("First step has no reward contrast")
    return checkpoint


def baseline_running(pid, baseline):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return b"baseline" in command and str(baseline).encode() in command


def training_running(pid, step):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return b"step" in command and str(step).encode() in command


def seconds_left(deadline):
    remaining = (deadline - datetime.datetime.now(datetime.UTC)).total_seconds()
    if remaining <= 0:
        raise TimeoutError("Overall experiment window expired")
    return remaining


def record_status(out, state):
    atomic_json(
        out / "status.json",
        {**state, "updated_at_utc": datetime.datetime.now(datetime.UTC).isoformat()},
    )


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
