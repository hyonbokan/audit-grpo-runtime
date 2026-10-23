"""The training controller: visit configs, visit records, backups, identity and evaluation."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import controller
import panels
import retention
import schedules
from benchmark_free_text import evaluation_row_ids
from cycle_contract import validate_config
from testing.cycles import schedule_cycle

WEEKEND = {"schedule_id": "weekend-grpo-20261009"}


def test_reduced_schedule_preserves_explicit_order_and_excludes_held_out_rows():
    selected = [f"project-{number}" for number in range(8)]
    config = {**WEEKEND, "training_row_ids": selected, "epochs": 1, "seed": 3407}
    assert controller.reviewed_training_order(config, [*selected, "unused"]) == selected
    for invalid in (
        [*selected, "unused"],
        [*selected[:-1], "held-out"],
        [*selected[:-1], selected[0]],
    ):
        with pytest.raises(ValueError):
            controller.reviewed_training_order(dict(config, training_row_ids=invalid), selected)


def test_next_group_reserves_measured_group_time_and_stops_after_repeated_truncation():
    state = {
        "visits": [{"seconds": 3600, "all_truncated": False}],
        "current_step": 2,
        "completion_gate": True,
    }
    assert controller.can_start_group(state, 5000, WEEKEND) is True
    assert controller.can_start_group(state, 4000, WEEKEND) is False
    state["visits"] = [{"seconds": 3600, "all_truncated": True}] * 2
    assert controller.can_start_group(state, 50000, WEEKEND) is False


def recorded(truncated, findings=5.0):
    return {
        "seconds": 3600,
        "all_truncated": False,
        "truncated_draws": truncated,
        "mean_findings": findings,
    }


def test_training_stops_when_over_a_tenth_of_the_last_twenty_draws_are_cut_off():
    state = {"current_step": 2}
    # Five groups are twenty draws; three cut-off draws are 15%, two are exactly 10%.
    state["visits"] = [recorded(1), recorded(1), recorded(1), recorded(0), recorded(0)]
    assert controller.can_start_group(state, 90000, WEEKEND) is False
    state["visits"] = [recorded(1), recorded(1), recorded(0), recorded(0), recorded(0)]
    assert controller.can_start_group(state, 90000, WEEKEND) is True


def test_a_short_history_cannot_trip_the_cut_off_rate():
    state = {"current_step": 2, "visits": [recorded(2), recorded(1)]}
    assert controller.can_start_group(state, 90000, WEEKEND) is True


def test_old_cut_off_draws_leave_the_window():
    state = {"current_step": 2}
    state["visits"] = [recorded(3), recorded(0), recorded(0), recorded(0), recorded(0), recorded(0)]
    assert controller.can_start_group(state, 90000, WEEKEND) is True


def test_failed_visits_carry_no_draws_and_do_not_fill_the_window():
    failed = {"seconds": 10, "all_truncated": False, "decision": "failed"}
    state = {"current_step": 2, "visits": [recorded(4), failed]}
    assert controller.can_start_group(state, 90000, WEEKEND) is True


def test_training_stops_when_reports_collapse_to_almost_no_findings():
    state = {"current_step": 2, "visits": [recorded(0, 2.9)] * 5}
    assert controller.can_start_group(state, 90000, WEEKEND) is False
    state["visits"] = [recorded(0, 2.9)] * 4 + [recorded(0, 3.0)]
    assert controller.can_start_group(state, 90000, WEEKEND) is True
    state["visits"] = [recorded(0, 2.0)] * 4 + [recorded(0, None)]
    assert controller.can_start_group(state, 90000, WEEKEND) is True


def test_checkpoint_record_does_not_count_a_skipped_visit_as_learning(tmp_path):
    state = {
        "visits": [],
        "current_step": 2,
        "current_checkpoint": "cp2",
        "current_adapter_sha256": "a" * 64,
        "completion_gate": False,
    }
    (tmp_path / "group-decision.json").write_text(
        json.dumps({"decision": "skip_no_reward_contrast"})
    )
    (tmp_path / "launcher-exit.txt").write_text("0\n")
    (tmp_path / "cycle-seconds.txt").write_text("5000\n")
    (tmp_path / "rollout").mkdir()
    (tmp_path / "rollout/samples.json").write_text(
        json.dumps([{"finish_reason": "length", "final_text": ""} for _ in range(4)])
    )
    (tmp_path / "grading").mkdir()
    (tmp_path / "grading/rewards.json").write_text("[0,0,0,0]")
    (tmp_path / "grading/metrics.json").write_text(json.dumps({"mean_findings": None}))
    controller.record_visit(state, tmp_path)
    assert state["current_step"] == 2
    assert state["completion_gate"] is False
    assert state["visits"][0]["all_truncated"] is True
    assert state["visits"][0]["truncated_draws"] == 4
    assert state["visits"][0]["mean_findings"] is None


def test_unaccepted_saved_update_cannot_advance_the_cursor(tmp_path):
    state = {"visits": [], "current_step": 2}
    (tmp_path / "launcher-exit.txt").write_text("1\n")
    with pytest.raises(ValueError, match="failed"):
        controller.record_visit(state, tmp_path)
    assert state == {"visits": [], "current_step": 2}


def test_saved_schedule_retains_remaining_updates_and_advances_the_cursor():
    config = {
        "initial_step": 2,
        "target_step": 3,
        "learning_rate": 5e-6,
        "continuation_schedule": {
            "kind": "finite_weekend",
            "maximum_step": 512,
            "cursor": 0,
            "order_sha256": "a" * 64,
        },
    }
    result = schedules.schedule_after_update(config)
    assert result["updates_remaining"] == 509
    assert result["cursor"] == 1
    assert result["order_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "launcher", ["launch-cycle.sh", "launch-evaluation.sh", "launch-benchmark-suite.sh"]
)
def test_launcher_verifies_fresh_results_on_ephemeral_disk_before_any_worker(tmp_path, launcher):
    code = tmp_path / "runtime/training-code"
    code.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name(launcher), code / launcher)
    shutil.copyfile(Path(__file__).with_name("launch-lib.sh"), code / "launch-lib.sh")
    results = tmp_path / "runtime/runs"
    results.mkdir()
    disk = tmp_path / "ephemeral/run"
    disk.mkdir(parents=True)
    (results / "run").symlink_to(disk, target_is_directory=True)
    (disk / "config.json").write_text("{}")
    (disk / "SOURCE_SHA256SUMS").write_text("a" * 64 + "  missing-source.py\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    probe = binary / "nvidia-smi"
    probe.write_text("#!/bin/sh\nexit 0\n")
    probe.chmod(0o755)
    docker = binary / "docker"
    docker.write_text(f"#!/bin/sh\ntouch {binary / 'docker-called'}\nexit 42\n")
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(code / launcher), str(results / "run"), str(results / "run/config.json")],
        capture_output=True,
        env={**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"]},
    )
    assert result.returncode != 0  # Missing source must fail before Docker.
    assert "missing-source.py: FAILED" in (disk / "source-verification.log").read_text()
    assert not (binary / "docker-called").exists()


def test_evaluation_freezes_the_latest_actual_step(tmp_path):
    state = {
        "current_step": 7,
        "current_checkpoint": "/results/cp7",
        "current_adapter_sha256": "d" * 64,
    }
    assert panels.trained_case(state) == {
        "label": "checkpoint-7",
        "initial_step": 7,
        "checkpoint": "/results/cp7",
        "initial_adapter_sha256": "d" * 64,
    }


def test_incomplete_cached_merge_is_rejected_before_serving(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "MERGE_MANIFEST.json").write_text(
        json.dumps({"status": "failed", "adapter_sha256": "a" * 64, "files": {}})
    )
    with pytest.raises(ValueError):
        panels.verify_merge(merged, checkpoint)


def test_failure_policy_defaults_to_resume_then_skip_and_rejects_unknown_keys():
    default = controller.failure_policy({})
    assert default["visit_attempts"] > 1 and default["skip_failed_visits"] is True
    strict = {"failure_policy": {"visit_attempts": 1, "skip_failed_visits": False}}
    assert controller.failure_policy(strict)["skip_failed_visits"] is False
    for invalid in ({"visit_atempts": 2}, {"visit_attempts": 0}, {"skip_failed_visits": 1}):
        with pytest.raises(ValueError):
            controller.failure_policy({"failure_policy": invalid})


def test_failed_visit_keeps_checkpoint_and_repeated_failures_stop_training(tmp_path):
    state = {"visits": [], "current_step": 3, "current_checkpoint": "cp3"}
    (tmp_path / "launcher-exit.txt").write_text("1\n")
    (tmp_path / "last-phase.txt").write_text("update\n")
    controller.record_failed_visit(state, tmp_path, attempts=2)
    assert state["current_step"] == 3 and state["current_checkpoint"] == "cp3"
    assert state["visits"][0] | {"root": None} == {
        "root": None,
        "decision": "failed",
        "attempts": 2,
        "last_phase": "update",
        "exit_status": "1",
        "seconds": 0,
        "all_truncated": False,
    }
    assert controller.can_start_group(state, 50000, WEEKEND) is True
    controller.record_failed_visit(state, tmp_path, attempts=2)
    assert controller.can_start_group(state, 50000, WEEKEND) is False


def test_next_cycle_resumes_an_unrecorded_pending_visit(tmp_path, monkeypatch):
    roomy_disk(monkeypatch, 1000)
    pending = str(tmp_path / "run-visit-002")
    (tmp_path / "state.json").write_text(json.dumps({"visits": [{"root": "run-visit-001"}]}))
    (tmp_path / "pending-training.json").write_text(json.dumps({"root": pending, "row_id": "r"}))
    assert controller.next_cycle(run_config(tmp_path), tmp_path) == pending


# --- Visits and identity

DAY = "medium-day-20261010"


def maximum_step():
    return schedules.schedule(DAY)["training"]["maximum_step"]


def run_config(tmp_path, **changes):
    """A run config whose result directories land under tmp_path."""
    (tmp_path / "run").mkdir(exist_ok=True)
    (tmp_path / "run/SOURCE_SHA256SUMS").write_text("frozen\n")
    for name in ("artifacts", "runtime/runs"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": "run",
        "schedule_id": DAY,
        "training_cutoff": schedules.schedule(DAY)["training_cutoff"],
        "cycle": schedule_cycle(DAY),
        "initial_checkpoint": "/results/start/checkpoint-3",
        "initial_adapter_sha256": "0" * 64,
        "artifact_host": str(tmp_path / "artifacts"),
        "runtime_host": str(tmp_path / "runtime"),
        "run_host": str(tmp_path / "run"),
    }
    return {**config, **changes}


def write_requests(out, row_id, prompt_tokens=18625, context=262144):
    manifest = {
        row_id: {
            "container_path": f"/results/{out.name}/requests/{row_id}.json",
            "prompt_tokens": prompt_tokens,
            "context": context,
        }
    }
    (out / "requests-manifest.json").write_text(json.dumps(manifest))


def test_next_cycle_builds_the_next_single_step_from_the_accepted_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "seconds_until", lambda _deadline: 40 * 3600)
    roomy_disk(monkeypatch, 1000)
    config = run_config(tmp_path)
    out = tmp_path / "runtime/runs/run"
    out.mkdir()
    state = {
        "visits": [{"root": "v1", "seconds": 1200, "all_truncated": False}],
        "current_step": 5,
        "current_checkpoint": "/results/run-visit-001/update/checkpoint-5",
        "current_adapter_sha256": "5" * 64,
        "completion_gate": True,
    }
    (out / "state.json").write_text(json.dumps(state))
    (out / "schedule.json").write_text(json.dumps(["first", "second"]))
    write_requests(out, "second")
    root = Path(controller.next_cycle(config, out))
    assert root == tmp_path / "runtime/runs/run-visit-002"
    written = json.loads((root / "config.json").read_text())
    assert written["row_id"] == "second"
    assert (written["initial_step"], written["target_step"]) == (5, 6)
    assert written["checkpoint"] == state["current_checkpoint"]
    assert written["initial_adapter_sha256"] == "5" * 64
    assert written["sampling"]["max_tokens"] == schedules.schedule(DAY)["maximum_context"] - 18625
    assert written["continuation_schedule"]["cursor"] == 1
    assert written["continuation_schedule"]["maximum_step"] == maximum_step()
    assert (root / "SOURCE_SHA256SUMS").read_text() == "frozen\n"
    assert json.loads((out / "pending-training.json").read_text()) == {
        "root": str(root),
        "row_id": "second",
    }
    validate_config(written)  # The controller's output satisfies the cycle contract.


def test_next_cycle_stops_at_the_step_cap_even_with_time_left(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "seconds_until", lambda _deadline: 40 * 3600)
    out = tmp_path / "out"
    out.mkdir()
    state = {"visits": [], "current_step": maximum_step(), "completion_gate": True}
    (out / "state.json").write_text(json.dumps(state))
    (out / "schedule.json").write_text(json.dumps(["first"]))
    config = {"schedule_id": DAY, "training_cutoff": "unused"}
    assert controller.next_cycle(config, out) == "complete"
    assert json.loads((out / "training-stop.json").read_text())["visits"] == 0
    assert not (out / "pending-training.json").exists()


def accepted_visit(root, *, global_step=3, approved=True, decision="update"):
    root.mkdir()
    (root / "launcher-exit.txt").write_text("0\n")
    (root / "cycle-seconds.txt").write_text("1500\n")
    (root / "group-decision.json").write_text(json.dumps({"decision": decision}))
    (root / "rollout").mkdir()
    draws = [
        {"finish_reason": "stop", "final_text": "{}", "completion_ids": [1, 2]} for _ in range(4)
    ]
    (root / "rollout/samples.json").write_text(json.dumps(draws))
    (root / "grading").mkdir()
    (root / "grading/rewards.json").write_text("[1, 0, 0, 0]")
    (root / "grading/metrics.json").write_text(json.dumps({"mean_findings": 6.5}))
    (root / "cycle-result.json").write_text(json.dumps({"decision": {"approved": approved}}))
    (root / "update").mkdir()
    receipt = {"global_step": global_step, "adapter_sha256": "3" * 64}
    (root / "update/receipt.json").write_text(json.dumps(receipt))
    return root


def test_accepted_update_advances_the_checkpoint_to_the_receipts_weights(tmp_path):
    state = {"visits": [], "current_step": 2, "completion_gate": False}
    root = accepted_visit(tmp_path / "run-visit-001")
    controller.record_visit(state, root)
    assert state["current_step"] == 3
    assert state["current_checkpoint"] == "/results/run-visit-001/update/checkpoint-3"
    assert state["current_adapter_sha256"] == "3" * 64
    assert state["completion_gate"] is True
    assert state["visits"][0]["completion_tokens"] == 8


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"global_step": 4}, "advance once"),
        ({"approved": False}, "Unaccepted"),
        ({"decision": "promote"}, "Unknown group decision"),
    ],
)
def test_unaccepted_or_skipped_steps_never_advance_the_checkpoint(tmp_path, changes, message):
    state = {"visits": [], "current_step": 2, "completion_gate": False}
    root = accepted_visit(tmp_path / "visit", **changes)
    with pytest.raises(ValueError, match=message):
        controller.record_visit(state, root)
    assert state == {"visits": [], "current_step": 2, "completion_gate": False}


def test_record_pending_stamps_the_row_and_persists_state(tmp_path):
    root = accepted_visit(tmp_path / "run-visit-001")
    (tmp_path / "state.json").write_text(
        json.dumps({"visits": [], "current_step": 2, "completion_gate": False})
    )
    pending = {"root": str(root), "row_id": "row-a"}
    (tmp_path / "pending-training.json").write_text(json.dumps(pending))
    controller.record_pending(tmp_path, controller.record_visit)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["visits"][0]["row_id"] == "row-a"
    assert saved["current_step"] == 3


@pytest.mark.parametrize("step,trained", [(3, "false"), (4, "true")])
def test_identity_reports_training_only_after_the_step_advanced(tmp_path, step, trained):
    config = run_config(tmp_path)
    state = {
        "current_step": step,
        "current_checkpoint": f"/results/cp{step}",
        "current_adapter_sha256": "a" * 64,
        "completion_gate": True,
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    lines = controller.phase_identity(config, tmp_path, 1)
    assert lines == [f"/results/cp{step}", panels.merged_path(config, state), str(step), trained]


def overlap_inputs(tmp_path, *, shared):
    """A one-row training corpus and a benchmark whose first project does or does not share it."""
    dataset = tmp_path / "dataset"
    (dataset / "data/private").mkdir(parents=True)
    content = "contract Vault {}"
    row = {"row_id": "new/train", "source_files": [{"path": "Vault.sol", "content": content}]}
    (dataset / "data/private/train.jsonl").write_text(json.dumps(row) + "\n")
    manifest = {"files": {}, "requests": {}, "roles": {"train": ["new/train"]}}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    assets = tmp_path / "assets"
    assets.mkdir()
    sources = []
    for index, project in enumerate(evaluation_row_ids()):
        lane, repository = project.split("/")
        text = content if shared and index == 0 else f"unrelated {index}"
        digest = hashlib.sha256(text.encode()).hexdigest()
        sources.append(
            {"lane": lane, "repository": repository, "scope_path": "A.sol", "sha256": digest}
        )
    (assets / "BENCHMARK_MANIFEST.json").write_text(json.dumps({"sources": sources}))
    return dataset, assets


def prepare_config(tmp_path, dataset, assets, **changes):
    return run_config(
        tmp_path,
        schedule_id=WEEKEND["schedule_id"],
        training_cutoff=schedules.scope(WEEKEND)["training_cutoff"],
        dataset_host=str(dataset),
        dataset_manifest_sha256=hashlib.sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        benchmark_assets=str(assets),
        benchmark_manifest_sha256=hashlib.sha256(
            (assets / "BENCHMARK_MANIFEST.json").read_bytes()
        ).hexdigest(),
        epochs=1,
        seed=1,
        training_row_ids=["new/train"],
        **changes,
    )


def test_prepare_refuses_a_training_row_that_shares_a_file_with_an_evaluation_project(tmp_path):
    dataset, assets = overlap_inputs(tmp_path, shared=True)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match=r"new/train.*Vault\.sol.*solidity/munchables"):
        controller.prepare(prepare_config(tmp_path, dataset, assets), out)
    assert not (out / "state.json").exists()


def test_prepare_refuses_a_benchmark_manifest_that_is_not_the_pinned_one(tmp_path):
    dataset, assets = overlap_inputs(tmp_path, shared=False)
    config = {**prepare_config(tmp_path, dataset, assets), "benchmark_manifest_sha256": "0" * 64}
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="Benchmark manifest"):
        controller.prepare(config, out)


def test_prepare_starts_at_the_configured_checkpoint_without_learning(tmp_path):
    dataset, assets = overlap_inputs(tmp_path, shared=False)
    config = prepare_config(tmp_path, dataset, assets)
    out = tmp_path / "out"
    out.mkdir()
    controller.prepare(config, out)
    state = json.loads((out / "state.json").read_text())
    assert state["current_step"] == config["cycle"]["initial_step"]
    assert state["current_checkpoint"] == config["initial_checkpoint"]
    assert state["completion_gate"] is False
    assert json.loads((out / "schedule.json").read_text()) == ["new/train"]
    with pytest.raises(FileExistsError):
        controller.prepare(config, out)


# --- Disk retention


def roomy_disk(monkeypatch, free_gib):
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda _path: (0, 0, free_gib * 2**30))


def test_next_cycle_refuses_to_start_a_visit_below_the_free_space_minimum(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "seconds_until", lambda _deadline: 40 * 3600)
    minimum = schedules.schedule(DAY)["storage"]["minimum_free_gib"]
    roomy_disk(monkeypatch, minimum - 1)
    config = run_config(tmp_path)
    out = tmp_path / "runtime/runs/run"
    out.mkdir()
    state = {"visits": [], "current_step": 5, "completion_gate": False}
    (out / "state.json").write_text(json.dumps(state))
    (out / "schedule.json").write_text(json.dumps(["first"]))
    write_requests(out, "first")
    with pytest.raises(OSError, match=rf"{minimum - 1}\.0 GiB free, {minimum} GiB required"):
        controller.next_cycle(config, out)
    assert not (out / "pending-training.json").exists()
    assert not (tmp_path / "runtime/runs/run-visit-001").exists()


def test_prune_phase_appends_what_it_freed_to_the_runs_retention_record(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    visits = []
    for index, step in enumerate((4, 5, 6), start=1):
        root = tmp_path / f"run-visit-{index:03d}"
        checkpoint = root / f"update/checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "optimizer.pt").write_bytes(b"o" * 7)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"a")
        visits.append({"root": str(root), "decision": "update", "update": {"global_step": step}})
    state = {"visits": visits, "current_checkpoint": str(checkpoint)}
    (out / "state.json").write_text(json.dumps(state))
    lines = controller.phase_prune({"schedule_id": DAY}, out, 1)
    oldest = tmp_path / "run-visit-001/update/checkpoint-4"
    assert json.loads(lines[0]) == {"files": 1, "bytes": 7}
    record = json.loads((out / "retention.json").read_text())
    assert [entry["removed"] for entry in record] == [
        [{"path": str(oldest / "optimizer.pt"), "bytes": 7}]
    ]
    assert (oldest / "adapter_model.safetensors").is_file()
    controller.phase_prune({"schedule_id": DAY}, out, 1)
    assert len(json.loads((out / "retention.json").read_text())) == 2


def backup_receipt(out, label, visits, status="complete_resumable_archive_download_verified"):
    path = out / f"backup-{label}-receipt.json"
    path.write_text(json.dumps({"status": status, "visits": visits}))


def test_a_visit_backup_is_due_once_the_interval_of_new_visits_has_passed(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    interval = schedules.scope({"schedule_id": DAY})["storage"]["visit_backup_interval"]
    backup_receipt(out, "initial", 0)
    due = []
    for count in range(1, 2 * interval + 2):
        (out / "state.json").write_text(json.dumps({"visits": [{}] * count}))
        due.append(controller.phase_backup_due({"schedule_id": DAY}, out, 1) == ["true"])
        if due[-1]:
            backup_receipt(out, f"visit-{count:03d}", count)
    assert [count for count, value in enumerate(due, start=1) if value] == [interval, 2 * interval]


def test_a_failed_visit_backup_leaves_the_backup_due(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    backup_receipt(out, "initial", 0)
    backup_receipt(out, "visit-005", 5, status="failed")
    (out / "state.json").write_text(json.dumps({"visits": [{}] * 6}))
    assert controller.phase_backup_due({"schedule_id": DAY}, out, 1) == ["true"]


def write_merge(merged, adapter_sha256):
    merged.mkdir(parents=True)
    manifest = {"status": "merged_and_verified", "adapter_sha256": adapter_sha256, "files": {}}
    (merged / "MERGE_MANIFEST.json").write_text(json.dumps(manifest))
    (merged / "model.safetensors").write_bytes(b"w" * 50)


def test_release_final_merge_deletes_the_evaluated_checkpoints_merge(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    state = {"current_step": 6, "current_adapter_sha256": "6" * 64}
    (out / "state.json").write_text(json.dumps(state))
    config = {"schedule_id": DAY, "run_id": "run"}
    merged = Path(panels.merged_path(config, state))
    monkeypatch.setattr(controller, "merged_path", lambda *_: str(tmp_path / merged.name))
    write_merge(tmp_path / merged.name, "6" * 64)
    lines = controller.phase_release_final_merge(config, out, 1)
    assert not (tmp_path / merged.name).exists()
    assert json.loads(lines[0])["bytes"] == 50 + 132
    assert (out / "merge-receipts" / f"{merged.name}.json").is_file()


def test_verify_final_merge_accepts_only_the_current_checkpoints_unchanged_merge(
    tmp_path, monkeypatch
):
    out = tmp_path / "run"
    out.mkdir()
    checkpoint = tmp_path / "checkpoint-6"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    digest = hashlib.sha256(b"adapter").hexdigest()
    state = {
        "current_step": 6,
        "current_checkpoint": str(checkpoint),
        "current_adapter_sha256": digest,
    }
    (out / "state.json").write_text(json.dumps(state))
    merged = tmp_path / "merged"
    monkeypatch.setattr(controller, "merged_path", lambda *_: str(merged))
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"weights")
    files = {"model.safetensors": {"merged_sha256": hashlib.sha256(b"weights").hexdigest()}}
    manifest = {"status": "merged_and_verified", "adapter_sha256": digest, "files": files}
    (merged / "MERGE_MANIFEST.json").write_text(json.dumps(manifest))
    config = {"schedule_id": DAY, "run_id": "run"}
    assert controller.phase_verify_final_merge(config, out, 1) == []
    (merged / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match=r"changed: model\.safetensors"):
        controller.phase_verify_final_merge(config, out, 1)


def write_frozen_base_runs(results, *, skip=None):
    """A finished base-only free-text run per project, as the schedule names them."""
    from benchmark_free_text import evaluation_row_ids
    from testing.benchmark import free_text_configuration

    pattern = schedules.scope({"schedule_id": DAY})["evaluation_base_reference"]
    hashes = {}
    for row_id in evaluation_row_ids():
        name = row_id.split("/", 1)[1]
        root = results / pattern.format(project=name)
        root.mkdir(parents=True)
        (root / "prepared.json").write_text(json.dumps(free_text_configuration(row_id)))
        (root / "private-truth.json").write_text(json.dumps([{"issue": i} for i in range(3)]))
        if name != skip:
            comparison = json.dumps({"draws": {"base": []}, "project": name})
            (root / "comparison.json").write_text(comparison)
            hashes[name] = hashlib.sha256(comparison.encode()).hexdigest()
    return pattern, hashes


def final_evaluation_run(tmp_path):
    config = run_config(
        tmp_path, judge_ledgers={"evaluation": "/results/run/evaluation-ledger.json"}
    )
    out = tmp_path / "runtime/runs/run"
    out.mkdir()
    state = {
        "current_step": 7,
        "current_checkpoint": "/results/cp7",
        "current_adapter_sha256": "d" * 64,
    }
    (out / "state.json").write_text(json.dumps(state))
    return config, out


def test_final_evaluation_runs_only_the_trained_arm_against_the_frozen_base(tmp_path):
    from benchmark_free_text import validate_config

    config, out = final_evaluation_run(tmp_path)
    pattern, hashes = write_frozen_base_runs(tmp_path / "runtime/runs")
    reviewed = schedules.scope(config)
    state = json.loads((out / "state.json").read_text())
    [suite_root] = panels.evaluation_configs(config, out)
    suite = json.loads((Path(suite_root) / "config.json").read_text())
    assert suite["output_format"] == "free_text" and suite["trained_label"] == "checkpoint-7"
    assert [p["name"] for p in suite["projects"]] == list(hashes)
    assert suite["acceptance"] == schedules.scope(config)["acceptance"]
    for project in suite["projects"]:
        name = project["name"]
        base_run = pattern.format(project=name)
        assert project["known_gt_count"] == 3
        assert project["base_run_root"] == f"/results/{base_run}"
        child = json.loads(
            (tmp_path / "runtime/runs" / project["run_id"] / "config.json").read_text()
        )
        frozen = json.loads((tmp_path / "runtime/runs" / base_run / "prepared.json").read_text())
        validate_config(child)
        assert child["cases"] == [panels.trained_case(state)]
        assert child["generation_order"] == ["checkpoint-7"]
        assert child["base_reference"] == {
            "run_root": f"/results/{base_run}",
            "comparison_sha256": hashes[name],
        }
        for key in ("sampling", "engine", "template_options", "benchmark_judge", "row_id"):
            assert child[key] == frozen[key]
        assert child["judge_operation_cap_usd"] == reviewed["evaluation_judge_cap_usd"]
        assert child["absolute_deadline"] == reviewed["final_deadline"]
        assert child["merged_model_path"] == panels.merged_path(config, state)


def test_final_evaluation_needs_every_frozen_base_before_writing_anything(tmp_path):
    config, out = final_evaluation_run(tmp_path)
    write_frozen_base_runs(tmp_path / "runtime/runs", skip="tigris")
    with pytest.raises(FileNotFoundError, match=r"comparison\.json"):
        panels.evaluation_configs(config, out)
    assert not list((tmp_path / "artifacts").iterdir())


@pytest.mark.parametrize(
    ("key", "value"), [("template_options", {"reasoning_effort": "medium"}), ("sampling", {"n": 2})]
)
def test_final_evaluation_refuses_frozen_bases_that_differ_in_effort_or_draws(tmp_path, key, value):
    config, out = final_evaluation_run(tmp_path)
    pattern, _ = write_frozen_base_runs(tmp_path / "runtime/runs")
    prepared = tmp_path / "runtime/runs" / pattern.format(project="phi") / "prepared.json"
    frozen = json.loads(prepared.read_text())
    prepared.write_text(json.dumps({**frozen, key: {**frozen[key], **value}}))
    with pytest.raises(ValueError, match="Frozen base runs differ"):
        panels.evaluation_configs(config, out)
    assert not (out / "final-identity.json").exists()
