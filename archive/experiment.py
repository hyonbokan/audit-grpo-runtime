"""Expanded training, fixed validation selection and locked final-test orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def validation_actions(config: dict, state: dict, *, finished: bool = False) -> list[str]:
    history = state["validation_history"]
    visits = len(state["visits"])
    due = finished or (visits > 0 and visits % config["validation_interval"] == 0)
    if not history:
        if not due and not (visits == 0 and config.get("baseline_validation_first", True)):
            return []
        return (
            ["checkpoint-2", "current"]
            if state["current_checkpoint"] != config.get("initial_checkpoint", "cp2")
            else ["checkpoint-2"]
        )
    if due and history[-1]["checkpoint"] != state["current_checkpoint"]:
        return ["current"]
    return []


def training_order(rows: list[str], *, epochs: int, seed: int) -> list[str]:
    if not rows or len(set(rows)) != len(rows) or not 1 <= epochs <= 2:
        raise ValueError("One or two finite shuffled passes of unique rows required")
    rng = random.Random(seed)
    result = []
    for _ in range(epochs):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        result.extend(shuffled)
    return result


def group_decision(rewards: list[float]) -> str:
    if len(rewards) != 4 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in rewards
    ):
        raise ValueError("Four finite known grades required")
    return "update" if len(set(rewards)) > 1 else "skip_no_reward_contrast"


def aggregate_panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty unique project panel required")
    for row in rows:
        for key in ("exact_coverage", "partial_coverage", "format_validity"):
            if (
                type(row[key]) not in (int, float)
                or not math.isfinite(row[key])
                or not 0 <= row[key] <= 1
            ):
                raise ValueError("Unknown or invalid project metric")
    return {
        "row_ids": sorted(r["row_id"] for r in rows),
        "projects": rows,
        "unknown_draws": sum(r["unknown_draws"] for r in rows),
        "macro_exact_coverage": sum(r["exact_coverage"] for r in rows) / len(rows),
        "macro_partial_coverage": sum(r["partial_coverage"] for r in rows) / len(rows),
        "macro_format_validity": sum(r["format_validity"] for r in rows) / len(rows),
    }


def select_checkpoint(
    history: list[dict[str, Any]], *, minimum_improvement: float
) -> dict[str, Any]:
    if not history or minimum_improvement < 0:
        raise ValueError("Baseline validation and nonnegative minimum improvement required")
    identity = history[0]["panel"]["row_ids"]
    best = history[0]
    for entry in history:
        panel = entry["panel"]
        if panel["row_ids"] != identity or panel["unknown_draws"]:
            raise ValueError("Validation panel differs or contains unknown grades")
        metric = panel["macro_exact_coverage"]
        if not math.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError("Invalid validation selection metric")
        if metric > best["panel"]["macro_exact_coverage"] + minimum_improvement:
            best = entry
    return best


def test_comparison(baseline: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    if (
        baseline["row_ids"] != selected["row_ids"]
        or baseline["unknown_draws"]
        or selected["unknown_draws"]
    ):
        raise ValueError("Matched known test projects required")
    left = {r["row_id"]: r for r in baseline["projects"]}
    right = {r["row_id"]: r for r in selected["projects"]}
    deltas = {
        identity: right[identity]["exact_coverage"] - row["exact_coverage"]
        for identity, row in left.items()
    }
    rng = random.Random(3407)
    values = list(deltas.values())
    boot = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(2000))
    return {
        "macro_exact_delta": selected["macro_exact_coverage"] - baseline["macro_exact_coverage"],
        "macro_partial_delta": selected["macro_partial_coverage"]
        - baseline["macro_partial_coverage"],
        "per_project_exact_delta": deltas,
        "descriptive_project_bootstrap_interval_95": [boot[50], boot[1949]],
        "uncertainty_scope": "Descriptive resampling of the small project panel; excludes judge/model measurement uncertainty",
        "quality_gain_established": False,
    }


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_dataset(config: dict[str, Any]) -> None:
    root = Path(config["dataset_host"])
    if (
        hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        != config["dataset_manifest_sha256"]
    ):
        raise ValueError("Frozen corpus manifest changed")
    for relative, digest in read(root / "manifest.json")["files"].items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"Frozen corpus file changed: {relative}")


def prepare(config: dict[str, Any], out: Path) -> None:
    if (out / "state.json").exists():
        raise FileExistsError("Experiment state must be fresh")
    if config["retry_limit"] != 0 or config["validation_interval"] < 1 or config["patience"] < 1:
        raise ValueError("Finite cadence and zero retries required")
    verify_dataset(config)
    dataset = Path(config["dataset_host"])
    if (
        hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
        != config["dataset_manifest_sha256"]
    ):
        raise ValueError("Frozen corpus manifest changed")
    manifest = read(dataset / "manifest.json")
    roles = manifest["roles"]
    if set(config["training_row_ids"]) != set(roles["train"]):
        raise ValueError("Training schedule does not cover the frozen training partition")
    if (
        not set(config["validation_row_ids"]).issubset(roles["calibration"])
        or len(config["validation_row_ids"]) < 2
    ):
        raise ValueError("At least two fixed validation projects required")
    if (
        not set(config["test_row_ids"]).issubset(roles["evaluation"])
        or len(config["test_row_ids"]) < 2
    ):
        raise ValueError("At least two untouched test projects required")
    if set(config["training_row_ids"]) & set(config["validation_row_ids"] + config["test_row_ids"]):
        raise ValueError("Training/held-out roles overlap")
    order = training_order(config["training_row_ids"], epochs=config["epochs"], seed=config["seed"])
    save(out / "schedule.json", order)
    save(
        out / "state.json",
        {
            "visits": [],
            "current_checkpoint": config["initial_checkpoint"],
            "current_step": 2,
            "current_adapter_sha256": config["initial_adapter_sha256"],
            "validation_history": [],
            "best_checkpoint": None,
            "training_complete": False,
        },
    )
    save(out / "prepared.json", config)


def next_training(config: dict[str, Any], out: Path) -> str:
    verify_dataset(config)
    state = read(out / "state.json")
    if len(state["visits"]) >= len(read(out / "schedule.json")):
        return "complete"
    if (
        datetime.fromisoformat(config["training_cutoff"]) - datetime.now(timezone.utc)
    ).total_seconds() < config["minimum_cycle_seconds"]:
        return "complete"
    if len(state["validation_history"]) >= config["patience"] + 1:
        best = select_checkpoint(
            state["validation_history"], minimum_improvement=config["minimum_improvement"]
        )
        if best in state["validation_history"][: -config["patience"]]:
            return "complete"
    identity = read(out / "schedule.json")[len(state["visits"])]
    request_info = read(Path(config["dataset_host"]) / "manifest.json")["requests"][identity]
    cycle = {
        **config["cycle"],
        "initial_step": state["current_step"],
        "target_step": state["current_step"] + 1,
        "checkpoint": state["current_checkpoint"],
        "initial_adapter_sha256": state["current_adapter_sha256"],
        "row_id": identity,
        "request": request_info["container_path"],
        "dataset_workspace": config["dataset_container"],
        "engine": {**config["cycle"]["engine"], "max_model_len": request_info["context"]},
        "skip_uninformative_groups": True,
    }
    root = (
        Path(config["runtime_host"])
        / "results"
        / f"{config['cycle'].get('run_id', 'full-training-20261009')}-visit-{len(state['visits']) + 1:03d}"
    )
    root.mkdir(exist_ok=False)
    cycle["seconds_limit"] = min(
        cycle["seconds_limit"],
        max(
            1,
            int(
                (
                    datetime.fromisoformat(config["training_cutoff"]) - datetime.now(timezone.utc)
                ).total_seconds()
            ),
        ),
    )
    save(root / "config.json", cycle)
    (root / "SOURCE_SHA256SUMS").write_bytes((out / "SOURCE_SHA256SUMS").read_bytes())
    save(out / "pending-training.json", {"root": str(root), "row_id": identity})
    return str(root)


def record_training(config: dict[str, Any], out: Path) -> dict[str, Any]:
    pending = read(out / "pending-training.json")
    root = Path(pending["root"])
    state = read(out / "state.json")
    cycle = read(root / "config.json")
    if (root / "launcher-exit.txt").read_text().strip() != "0":
        raise ValueError("Training cycle failed; preserve evidence and stop")
    decision = read(root / "group-decision.json")["decision"]
    visit = {
        **pending,
        "decision": decision,
        "initial_step": cycle["initial_step"],
        "rewards": read(root / "grading/rewards.json"),
        "metrics": read(root / "grading/metrics.json"),
        "completion_tokens": read(root / "rollout/receipt.json")["completion_tokens"],
    }
    if decision == "update":
        accepted = read(root / "cycle-result.json")
        if not accepted["decision"]["approved"]:
            raise ValueError("Unaccepted optimizer update cannot progress")
        receipt = read(root / "update/receipt.json")
        state.update(
            current_checkpoint=f"/results/{root.name}/update/checkpoint-{cycle['target_step']}",
            current_step=cycle["target_step"],
            current_adapter_sha256=receipt["adapter_sha256"],
        )
        visit["update"] = receipt
    state["visits"].append(visit)
    save(out / "state.json", state)
    cadence = (
        len(state["visits"]) % config["validation_interval"] == 0
        or len(state["visits"]) % len(config["training_row_ids"]) == 0
    )
    return {
        "validation_due": cadence,
        "completed_updates": sum(v["decision"] == "update" for v in state["visits"]),
        "consumed_rows": len(state["visits"]),
    }


def make_panel_configs(config: dict[str, Any], out: Path, role: str, label: str) -> list[str]:
    verify_dataset(config)
    state = read(out / "state.json")
    if role == "test":
        if not (out / "selection-lock.json").is_file():
            raise ValueError("Final test requires frozen checkpoint selection")
        lock = read(out / "selection-lock.json")
        if label == "selected":
            checkpoint, step, digest = lock["checkpoint"], lock["step"], lock["adapter_sha256"]
        elif label == "checkpoint-2":
            checkpoint, step, digest = (
                config["initial_checkpoint"],
                2,
                config["initial_adapter_sha256"],
            )
        elif label == "base":
            checkpoint, step, digest = None, 0, None
        else:
            raise ValueError("Unknown final test arm")
    elif label == "checkpoint-2":
        checkpoint, step, digest = config["initial_checkpoint"], 2, config["initial_adapter_sha256"]
    else:
        checkpoint, step, digest = (
            state["current_checkpoint"],
            state["current_step"],
            state["current_adapter_sha256"],
        )
    manifest = read(Path(config["dataset_host"]) / "manifest.json")
    roots = []
    for index, identity in enumerate(
        config[f"{'validation' if role == 'validation' else 'test'}_row_ids"]
    ):
        request = manifest["requests"][identity]
        key = (
            f"v{len(state['validation_history']):02d}" if role == "validation" else f"test-{label}"
        )
        root = (
            Path(config["runtime_host"])
            / "results"
            / f"{config['cycle'].get('run_id', 'full-training-20261009')}-{key}-p{index + 1}"
        )
        root.mkdir(exist_ok=False)
        case_label = "base" if label == "base" else label
        base = config["cycle"]
        panel = {
            **base,
            "evaluation_kind": "grpo_held_out_panel",
            "split": "calibration" if role == "validation" else "evaluation",
            "request": request["container_path"],
            "engine": {
                **base["engine"],
                "return_sampling_mask": False,
                "max_model_len": request["context"],
            },
            "row_id": identity,
            "dataset_workspace": config["dataset_container"],
            "workspace": "/workspace",
            "seconds_limit": config["panel_project_seconds"],
            "judge_seconds_limit": 1800,
            "cases": [
                {
                    "label": case_label,
                    "checkpoint": checkpoint,
                    "initial_step": step,
                    "initial_adapter_sha256": digest,
                }
            ],
        }
        for field in (
            "schedule_id",
            "checkpoint",
            "initial_adapter_sha256",
            "initial_step",
            "target_step",
        ):
            panel.pop(field, None)
        save(root / "config.json", panel)
        (root / "SOURCE_SHA256SUMS").write_bytes((out / "SOURCE_SHA256SUMS").read_bytes())
        roots.append(str(root))
    save(
        out / f"pending-{role}-{label}.json",
        {
            "roots": roots,
            "checkpoint": checkpoint,
            "step": step,
            "adapter_sha256": digest,
            "label": label,
        },
    )
    return roots


def record_panel(config: dict[str, Any], out: Path, role: str, label: str) -> dict[str, Any]:
    pending = read(out / f"pending-{role}-{label}.json")
    rows = []
    for root in map(Path, pending["roots"]):
        if (root / "launcher-exit.txt").read_text().strip() != "0":
            raise ValueError("Panel failed; checkpoint selection withheld")
        rows.append(read(root / "comparison.json")["metrics"])
    panel = aggregate_panel(rows)
    entry = {**pending, "panel": panel}
    if panel["unknown_draws"]:
        raise ValueError("Unknown panel grades cannot select a checkpoint")
    if role == "validation":
        state = read(out / "state.json")
        state["validation_history"].append(entry)
        state["best_checkpoint"] = select_checkpoint(
            state["validation_history"], minimum_improvement=config["minimum_improvement"]
        )
        save(out / "state.json", state)
    save(
        out
        / f"{role}-{label}-panel-{len(read(out / 'state.json')['validation_history']):02d}.json",
        entry,
    )
    return entry


def lock_selection(config: dict[str, Any], out: Path) -> dict[str, Any]:
    state = read(out / "state.json")
    best = select_checkpoint(
        state["validation_history"], minimum_improvement=config["minimum_improvement"]
    )
    if state["validation_history"][-1]["checkpoint"] != state["current_checkpoint"]:
        raise ValueError("Latest checkpoint must be validated before final selection")
    lock = {
        "checkpoint": best["checkpoint"],
        "step": best["step"],
        "adapter_sha256": best["adapter_sha256"],
        "panel": best["panel"],
        "criterion": "macro per-project mean four-draw exact known-finding coverage",
        "configuration_sha256": hashlib.sha256((out / "prepared.json").read_bytes()).hexdigest(),
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    if (out / "selection-lock.json").exists():
        raise FileExistsError("Checkpoint selection is already locked")
    save(out / "selection-lock.json", lock)
    state["training_complete"] = True
    save(out / "state.json", state)
    return lock


def final_report(config: dict[str, Any], out: Path) -> dict[str, Any]:
    verify_dataset(config)
    state = read(out / "state.json")
    lock = read(out / "selection-lock.json")
    if (
        lock["configuration_sha256"]
        != hashlib.sha256((out / "prepared.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Experiment changed after checkpoint selection")
    panels = {}
    for label in ("checkpoint-2", "selected", "base"):
        if label == "selected" and lock["step"] == 2:
            panels[label] = panels["checkpoint-2"]
            continue
        paths = list(out.glob(f"test-{label}-panel-*.json"))
        if len(paths) != 1:
            raise ValueError("Exactly one final test panel per distinct arm required")
        panels[label] = read(paths[0])["panel"]
    manifest = read(Path(config["dataset_host"]) / "manifest.json")
    report = {
        "status": "expanded_experiment_completed",
        "selected_checkpoint": lock,
        "dataset_counts": manifest["counts"],
        "training_visits": len(state["visits"]),
        "optimizer_updates": sum(v["decision"] == "update" for v in state["visits"]),
        "unique_training_rows_consumed": len({v["row_id"] for v in state["visits"]}),
        "planned_training_rows": len(config["training_row_ids"]),
        "planned_epochs": config["epochs"],
        "complete_passes": len(state["visits"]) // len(config["training_row_ids"]),
        "skipped_uninformative_groups": sum(
            v["decision"] == "skip_no_reward_contrast" for v in state["visits"]
        ),
        "validation_history": state["validation_history"],
        "test_panels": panels,
        "selected_vs_checkpoint2": test_comparison(panels["checkpoint-2"], panels["selected"]),
        "selected_vs_base": test_comparison(panels["base"], panels["selected"]),
        "test_selection_used": False,
        "regression_results_used_for_selection": False,
        "limits": "Sparse known labels; small fixed held-out panels. Partial corpus traversal is reported explicitly.",
    }
    save(out / "report.json", report)
    lines = [
        "# Expanded training experiment",
        "",
        f"Selected checkpoint: step {lock['step']}.",
        f"Training: {report['training_visits']} groups, {report['optimizer_updates']} optimizer updates, "
        f"{report['unique_training_rows_consumed']}/{report['planned_training_rows']} distinct rows consumed, "
        f"{report['complete_passes']} complete passes.",
        "",
        f"Test exact-coverage delta versus checkpoint 2: {report['selected_vs_checkpoint2']['macro_exact_delta']:+.2%}.",
        f"Test exact-coverage delta versus base: {report['selected_vs_base']['macro_exact_delta']:+.2%}.",
        "",
        "Checkpoint selection used fixed validation only. Test was run after locking selection.",
        "These small-panel, sparse-known-GT measurements do not establish broad quality improvement.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "next",
            "record",
            "panel",
            "record-panel",
            "lock",
            "group-decision",
            "report",
            "validation-actions",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=("validation", "test"))
    parser.add_argument("--label", default="checkpoint-2")
    parser.add_argument("--finished", action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    if args.phase == "validation-actions":
        print(
            "\n".join(
                validation_actions(config, read(args.out / "state.json"), finished=args.finished)
            )
        )
    elif args.phase == "prepare":
        prepare(config, args.out)
    elif args.phase == "next":
        print(next_training(config, args.out))
    elif args.phase == "record":
        result = record_training(config, args.out)
        print("validation_due" if result["validation_due"] else "continue")
    elif args.phase == "panel":
        print("\n".join(make_panel_configs(config, args.out, args.role, args.label)))
    elif args.phase == "record-panel":
        print(json.dumps(record_panel(config, args.out, args.role, args.label)))
    elif args.phase == "lock":
        print(json.dumps(lock_selection(config, args.out)))
    elif args.phase == "report":
        print(json.dumps(final_report(config, args.out)))
    else:
        rewards = read(args.out / "grading/rewards.json")
        decision = group_decision(rewards)
        save(
            args.out / "group-decision.json",
            {
                "decision": decision,
                "rewards": rewards,
                "row_id": config["row_id"],
                "initial_step": config["initial_step"],
            },
        )
        print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
