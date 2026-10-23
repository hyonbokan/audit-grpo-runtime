"""Multi-visit GRPO training: one accepted corrected cycle per visit, within a reviewed schedule.

`launch-training.sh` calls one phase per step and owns Docker, backups and deadlines.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from benchmark_free_text import evaluation_row_ids
from corpus_checks import check_benchmark_overlap
from frozen_requests import extend_request, rerender_request
from hf_backup import backed_up_visits
from panels import evaluation_configs, merged_path, verify_merge
from progress import training_order, verify_dataset
from retention import prune_visits, release_merge, require_free_space
from runtime_io import new_result_dir, read_json, save_json, seconds_until, sha256_file, utc_now
from schedules import scope

GROUP_SIZE = 4
# A visit starts only if this much time remains; later visits size it from recent visits.
VISIT_RESERVE_MINIMUM_SECONDS = 1800
VISIT_RESERVE_FACTOR = 1.25
VISIT_RESERVE_RECENT_VISITS = 3
VISIT_START_MARGIN_SECONDS = 300
# Leaves the controller time to record a visit before the training cutoff.
VISIT_DEADLINE_MARGIN_SECONDS = 120

# Defaults keep training through isolated failures: a failed visit is resumed once
# (completed phases are reused), then recorded as failed and skipped.
DEFAULT_FAILURE_POLICY = {
    "visit_attempts": 2,
    "skip_failed_visits": True,
    "maximum_consecutive_failed_visits": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for line in PHASES[args.phase](read_json(args.config), args.out, args.attempts):
        print(line)
    return 0


# --- CLI phases: each returns the lines the launcher reads from stdout


def phase_prepare(config: dict, out: Path, _attempts: int) -> list[str]:
    prepare(config, out)
    return []


def phase_failure_policy(config: dict, _out: Path, _attempts: int) -> list[str]:
    policy = failure_policy(config)
    return [str(policy["visit_attempts"]), str(int(policy["skip_failed_visits"]))]


def phase_next(config: dict, out: Path, _attempts: int) -> list[str]:
    return [next_cycle(config, out)]


def phase_record(config: dict, out: Path, _attempts: int) -> list[str]:
    state = record_pending(out, record_visit)
    summary = {
        "visits": len(state["visits"]),
        "step": state["current_step"],
        "completion_gate": state["completion_gate"],
    }
    return [json.dumps(summary)]


def phase_record_failure(config: dict, out: Path, attempts: int) -> list[str]:
    state = record_pending(out, lambda state, root: record_failed_visit(state, root, attempts))
    return [json.dumps(state["visits"][-1])]


def phase_backup_due(config: dict, out: Path, _attempts: int) -> list[str]:
    """Whether enough visits passed since the last verified backup to upload another."""
    interval = scope(config)["storage"]["visit_backup_interval"]
    if type(interval) is not int or interval < 1:
        raise ValueError("storage.visit_backup_interval must be an integer of at least 1")
    visits = len(read_json(out / "state.json")["visits"])
    return [str(visits - backed_up_visits(out) >= interval).lower()]


def phase_prune(config: dict, out: Path, _attempts: int) -> list[str]:
    """Free resume state and adapter copies no later phase reads; append the record."""
    state = read_json(out / "state.json")
    storage = scope(config)["storage"]
    removed = prune_visits(state, storage["resume_states_kept"], storage["resume_steps_pinned"])
    path = out / "retention.json"
    record = read_json(path) if path.is_file() else []
    record.append({"utc": utc_now().isoformat(), "removed": removed})
    save_json(path, record)
    return [json.dumps({"files": len(removed), "bytes": sum(e["bytes"] for e in removed)})]


def phase_verify_final_merge(config: dict, out: Path, _attempts: int) -> list[str]:
    """Refuse a cached final merge that is incomplete or holds other weights."""
    state = read_json(out / "state.json")
    verify_merge(Path(merged_path(config, state)), Path(state["current_checkpoint"]))
    return []


def phase_release_final_merge(config: dict, out: Path, _attempts: int) -> list[str]:
    merged = Path(merged_path(config, read_json(out / "state.json")))
    released = release_merge(merged, out / "merge-receipts" / f"{merged.name}.json")
    return [json.dumps({"merged_model_path": str(merged), "bytes": released and released["bytes"]})]


def phase_identity(config: dict, out: Path, _attempts: int) -> list[str]:
    """Checkpoint, merge path, step and whether training produced an accepted update."""
    state = read_json(out / "state.json")
    trained = state["completion_gate"] and state["current_step"] > config["cycle"]["initial_step"]
    return [
        state["current_checkpoint"],
        merged_path(config, state),
        str(state["current_step"]),
        str(trained).lower(),
    ]


def phase_evaluation(config: dict, out: Path, _attempts: int) -> list[str]:
    return evaluation_configs(config, out)


PHASES = {
    "prepare": phase_prepare,
    "failure-policy": phase_failure_policy,
    "next": phase_next,
    "record": phase_record,
    "record-failure": phase_record_failure,
    "backup-due": phase_backup_due,
    "prune": phase_prune,
    "identity": phase_identity,
    "evaluation": phase_evaluation,
    "verify-final-merge": phase_verify_final_merge,
    "release-final-merge": phase_release_final_merge,
}


# --- Preparation


def prepare(config: dict, out: Path) -> None:
    verify_dataset(config)
    failure_policy(config)
    if (out / "state.json").exists():
        raise FileExistsError("Training state must be fresh")
    reviewed = scope(config)
    if config["training_cutoff"] != reviewed["training_cutoff"]:
        raise ValueError("Training cutoff differs from the reviewed schedule")
    dataset = Path(config["dataset_host"])
    verify_no_benchmark_overlap(config, dataset)
    manifest = read_json(dataset / "manifest.json")
    effort = reviewed["training"]["policy_reasoning_effort"]
    policy, tokenizer = {}, None
    if effort == "medium":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["cycle"]["model_path"], local_files_only=True
        )
        rows = map(json.loads, (dataset / "data/policy/train.jsonl").read_text().splitlines())
        policy = {row["row_id"]: row["messages"] for row in rows}
    requests = {}
    for row_id, info in manifest["requests"].items():
        original = dataset / "requests" / Path(info["container_path"]).name
        request, parent = read_json(original), sha256_file(original)
        if row_id in policy:
            frozen = rerender_request(
                request,
                policy[row_id],
                tokenizer,
                effort,
                parent,
                maximum_context=reviewed["maximum_context"],
            )
        else:
            frozen = extend_request(request, parent, maximum_context=reviewed["maximum_context"])
        path = out / "requests" / original.name
        save_json(path, frozen)
        requests[row_id] = {
            "container_path": f"/results/{out.name}/requests/{original.name}",
            "context": frozen["max_model_len"],
            "prompt_tokens": len(frozen["prompt_ids"]),
            "request_sha256": sha256_file(path),
        }
    save_json(out / "requests-manifest.json", requests)
    save_json(out / "schedule.json", reviewed_training_order(config, manifest["roles"]["train"]))
    save_json(
        out / "state.json",
        {
            "visits": [],
            "current_step": config["cycle"]["initial_step"],
            "current_checkpoint": config["initial_checkpoint"],
            "current_adapter_sha256": config["initial_adapter_sha256"],
            "completion_gate": False,
            "started_utc": utc_now().isoformat(),
        },
    )


def verify_no_benchmark_overlap(config: dict, dataset: Path) -> None:
    """The rows this run may train never share a file with the evaluation projects."""
    manifest_path = Path(config["benchmark_assets"]) / "BENCHMARK_MANIFEST.json"
    if sha256_file(manifest_path) != config["benchmark_manifest_sha256"]:
        raise ValueError("Benchmark manifest identity differs")
    admitted = read_json(dataset / "manifest.json")["roles"]["train"]
    selected = set(config.get("training_row_ids", admitted))
    rows = [
        row
        for row in map(json.loads, (dataset / "data/private/train.jsonl").read_text().splitlines())
        if row["row_id"] in selected
    ]
    check_benchmark_overlap(rows, read_json(manifest_path), evaluation_row_ids())


def reviewed_training_order(config: dict, admitted: list[str]) -> list[str]:
    maximum = scope(config)["training"]["maximum_groups"]
    selected = config.get("training_row_ids", admitted)
    if (
        config["epochs"] != 1
        or not 0 < len(selected) <= maximum
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(admitted)
    ):
        raise ValueError(f"Schedule requires at most {maximum} unique admitted training rows")
    if "training_row_ids" in config:
        return list(selected)
    return training_order(selected, epochs=1, seed=config["seed"])


def failure_policy(config: dict) -> dict[str, Any]:
    policy = {**DEFAULT_FAILURE_POLICY, **config.get("failure_policy", {})}
    if set(policy) != set(DEFAULT_FAILURE_POLICY):
        raise ValueError(
            f"Unknown failure_policy keys: {sorted(set(policy) - set(DEFAULT_FAILURE_POLICY))}"
        )
    if type(policy["visit_attempts"]) is not int or not 1 <= policy["visit_attempts"] <= 3:
        raise ValueError("failure_policy.visit_attempts must be 1, 2 or 3")
    if type(policy["skip_failed_visits"]) is not bool:
        raise ValueError("failure_policy.skip_failed_visits must be true or false")
    if (
        type(policy["maximum_consecutive_failed_visits"]) is not int
        or policy["maximum_consecutive_failed_visits"] < 1
    ):
        raise ValueError("failure_policy.maximum_consecutive_failed_visits must be positive")
    return policy


# --- Visits


def next_cycle(config: dict, out: Path) -> str:
    """The next visit's cycle directory, or "complete" when the schedule or window ends."""
    state = read_json(out / "state.json")
    pending = out / "pending-training.json"
    if pending.is_file():
        root = read_json(pending)["root"]
        if root not in {visit["root"] for visit in state["visits"]}:
            # The controller stopped before recording this visit; resume it in place.
            require_visit_space(config, out)
            return root
    order = read_json(out / "schedule.json")
    remaining = seconds_until(config["training_cutoff"])
    if len(state["visits"]) >= len(order) or not can_start_group(state, remaining, config):
        save_json(
            out / "training-stop.json",
            {
                "remaining_seconds": remaining,
                "visits": len(state["visits"]),
                "completion_gate": state["completion_gate"],
                "utc": utc_now().isoformat(),
            },
        )
        return "complete"
    require_visit_space(config, out)
    identity = order[len(state["visits"])]
    request = read_json(out / "requests-manifest.json")[identity]
    cycle = copy.deepcopy(config["cycle"])
    cycle.update(
        row_id=identity,
        request=request["container_path"],
        prompt_token_count=request["prompt_tokens"],
        initial_step=state["current_step"],
        target_step=state["current_step"] + 1,
        checkpoint=state["current_checkpoint"],
        initial_adapter_sha256=state["current_adapter_sha256"],
        seconds_limit=min(cycle["seconds_limit"], int(remaining) - VISIT_DEADLINE_MARGIN_SECONDS),
        continuation_schedule={
            "kind": "finite_weekend",
            "maximum_step": scope(config)["training"]["maximum_step"],
            "training_cutoff": config["training_cutoff"],
            "order_sha256": sha256_file(out / "schedule.json"),
            "cursor": len(state["visits"]),
        },
    )
    cycle["sampling"]["max_tokens"] = request["context"] - request["prompt_tokens"]
    root = new_result_dir(config, f"{out.name}-visit-{len(state['visits']) + 1:03d}")
    save_json(root / "config.json", cycle)
    save_json(out / "pending-training.json", {"root": str(root), "row_id": identity})
    return str(root)


def require_visit_space(config: dict, out: Path) -> None:
    """Results, artifacts and merges must each have room before a visit writes to them."""
    paths = [out, Path(config["artifact_host"]), Path(scope(config)["merge_cache"])]
    require_free_space(paths, scope(config)["storage"]["minimum_free_gib"])


def can_start_group(state: dict, remaining_seconds: float, config: dict) -> bool:
    visits = state["visits"]
    reviewed = scope(config)
    failures = failure_policy(config)["maximum_consecutive_failed_visits"]
    if len(visits) >= failures and all(v.get("decision") == "failed" for v in visits[-failures:]):
        return False
    if state["current_step"] >= reviewed["training"]["maximum_step"]:
        return False
    count = reviewed["maximum_consecutive_truncated_groups"]
    if len(visits) >= count and all(v["all_truncated"] for v in visits[-count:]):
        return False
    if truncation_exceeded(visits, reviewed) or findings_collapsed(visits, reviewed):
        return False
    reserve = reviewed["initial_group_reserve_seconds"]
    if visits:
        recent = max(v["seconds"] for v in visits[-VISIT_RESERVE_RECENT_VISITS:])
        reserve = max(VISIT_RESERVE_MINIMUM_SECONDS, recent * VISIT_RESERVE_FACTOR)
    return remaining_seconds > reserve + VISIT_START_MARGIN_SECONDS


def truncation_exceeded(visits: list[dict], reviewed: dict) -> bool:
    """More than the reviewed fraction of the last window's draws hit the length cap."""
    groups = reviewed["truncation_window_draws"] // GROUP_SIZE
    window = [v["truncated_draws"] for v in visits if "truncated_draws" in v][-groups:]
    return len(window) == groups and sum(window) > (
        reviewed["maximum_truncated_draw_fraction"] * groups * GROUP_SIZE
    )


def findings_collapsed(visits: list[dict], reviewed: dict) -> bool:
    """Every recent group's reports list fewer findings than the reviewed minimum."""
    groups = reviewed["findings_window_groups"]
    window = [v["mean_findings"] for v in visits if "mean_findings" in v][-groups:]
    return len(window) == groups and all(
        mean is not None and mean < reviewed["minimum_mean_findings"] for mean in window
    )


def record_pending(out: Path, record: Any) -> dict:
    """Apply `record` to the pending visit and persist the state."""
    pending = read_json(out / "pending-training.json")
    state = read_json(out / "state.json")
    record(state, Path(pending["root"]))
    state["visits"][-1]["row_id"] = pending["row_id"]
    save_json(out / "state.json", state)
    return state


def record_visit(state: dict, root: Path) -> None:
    if (root / "launcher-exit.txt").read_text().strip() != "0":
        raise ValueError("Training cycle failed; preserve previous accepted checkpoint")
    decision = read_json(root / "group-decision.json")["decision"]
    samples = read_json(root / "rollout/samples.json")
    if len(samples) != GROUP_SIZE:
        raise ValueError("Complete four-answer group required")
    visit = {
        "root": str(root),
        "decision": decision,
        "seconds": int((root / "cycle-seconds.txt").read_text()),
        "all_truncated": all(s["finish_reason"] != "stop" for s in samples),
        "truncated_draws": sum(s["finish_reason"] == "length" for s in samples),
        "mean_findings": read_json(root / "grading/metrics.json")["mean_findings"],
        "rewards": read_json(root / "grading/rewards.json"),
        "completed_final_answers": sum(
            s["finish_reason"] == "stop" and bool(s["final_text"]) for s in samples
        ),
        "completion_tokens": sum(len(s.get("completion_ids", [])) for s in samples),
    }
    if decision == "update":
        accepted = read_json(root / "cycle-result.json")
        receipt = read_json(root / "update/receipt.json")
        if not accepted["decision"]["approved"] or not visit["completed_final_answers"]:
            raise ValueError("Unaccepted or incomplete training update cannot progress")
        if receipt["global_step"] != state["current_step"] + 1:
            raise ValueError("Optimizer counter must advance once")
        state.update(
            current_step=receipt["global_step"],
            current_checkpoint=f"/results/{root.name}/update/checkpoint-{receipt['global_step']}",
            current_adapter_sha256=receipt["adapter_sha256"],
            completion_gate=True,
        )
        visit["update"] = receipt
    elif decision != "skip_no_reward_contrast":
        raise ValueError(f"Unknown group decision: {decision!r}")
    state["visits"].append(visit)


def record_failed_visit(state: dict, root: Path, attempts: int) -> None:
    """Keep the accepted checkpoint and move on; the failed visit stays as evidence."""

    def text(name: str) -> str:
        path = root / name
        return path.read_text().strip() if path.is_file() else "unknown"

    seconds = root / "cycle-seconds.txt"
    state["visits"].append(
        {
            "root": str(root),
            "decision": "failed",
            "attempts": attempts,
            "last_phase": text("last-phase.txt"),
            "exit_status": text("launcher-exit.txt"),
            "seconds": int(seconds.read_text()) if seconds.is_file() else 0,
            "all_truncated": False,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
