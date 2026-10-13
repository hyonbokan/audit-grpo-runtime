"""Validation panels during training and the locked final evaluation suite."""

from __future__ import annotations

import copy
from pathlib import Path

from progress import aggregate_panel, select_checkpoint
from runtime_io import new_result_dir, read_json, save_json, seconds_until, sha256_file, utc_now
from schedules import scope

# Cycle settings that describe an optimizer update, not an evaluation arm.
UPDATE_ONLY_KEYS = (
    "schedule_id",
    "initial_step",
    "target_step",
    "checkpoint",
    "initial_adapter_sha256",
    "continuation_schedule",
)
LORA_ENGINE_KEYS = ("lora_dtype", "max_loras", "max_lora_rank")


def evaluation_cases(state: dict) -> list[dict]:
    return [
        {"label": "base", "initial_step": 0},
        {
            "label": f"checkpoint-{state['current_step']}",
            "initial_step": state["current_step"],
            "checkpoint": state["current_checkpoint"],
            "initial_adapter_sha256": state["current_adapter_sha256"],
        },
    ]


def merged_path(config: dict, state: dict) -> str:
    step, digest = state["current_step"], state["current_adapter_sha256"][:12]
    return f"{scope(config)['merge_cache']}/{config['run_id']}-step-{step}-{digest}"


def verify_merge(merged: Path, checkpoint: Path) -> None:
    manifest = read_json(merged / "MERGE_MANIFEST.json")
    if (
        manifest.get("status") != "merged_and_verified"
        or manifest.get("adapter_sha256") != sha256_file(checkpoint / "adapter_model.safetensors")
        or not manifest.get("files")
    ):
        raise ValueError("Cached inference merge is incomplete or identifies different weights")
    for relative, evidence in manifest["files"].items():
        if sha256_file(merged / relative) != evidence["merged_sha256"]:
            raise ValueError(f"Cached merged file changed: {relative}")


# --- Validation during training


def validation_due(
    state: dict, reviewed: dict, *, now_epoch: float, remaining_seconds: float
) -> bool:
    return (
        state["completion_gate"]
        and state["current_step"] > 2
        and now_epoch - state["last_validation_epoch"] >= reviewed["validation_interval_seconds"]
        and remaining_seconds > reviewed["validation_minimum_remaining_seconds"]
    )


def validation_configs(config: dict, out: Path) -> list[str]:
    reviewed = scope(config)
    state = read_json(out / "state.json")
    if not validation_due(
        state,
        reviewed,
        now_epoch=utc_now().timestamp(),
        remaining_seconds=seconds_until(config["training_cutoff"]),
    ):
        return []
    requests = read_json(out / "requests-manifest.json")
    cases = [evaluation_cases(state)[-1]]
    if not state["validation_history"]:
        initial = config["cycle"]["initial_step"]
        cases.insert(
            0,
            {
                "label": f"checkpoint-{initial}",
                "initial_step": initial,
                "checkpoint": config["initial_checkpoint"],
                "initial_adapter_sha256": config["initial_adapter_sha256"],
            },
        )
    folders = []
    for case in cases:
        identity = {
            "current_step": case["initial_step"],
            "current_adapter_sha256": case["initial_adapter_sha256"],
        }
        for index, row_id in enumerate(config["validation_row_ids"]):
            request = requests[row_id]
            history = len(state["validation_history"])
            name = f"{out.name}-validation-{history:02d}-{case['label']}-{index}"
            root = new_result_dir(config, name)
            panel = copy.deepcopy(config["cycle"])
            for key in UPDATE_ONLY_KEYS:
                panel.pop(key, None)
            panel.update(
                run_id=name,
                weekend_schedule_id=reviewed["schedule_id"],
                evaluation_kind="grpo_held_out_panel",
                split="calibration",
                row_id=row_id,
                request=request["container_path"],
                prompt_token_count=request["prompt_tokens"],
                cases=[case],
                merged_model_path=merged_path(config, identity),
                source_plan=f"/results/{out.name}/validation-source-plan.json",
                seconds_limit=reviewed["validation_seconds_limit"],
                judge_operation_cap_usd=reviewed["validation_judge_cap_usd"],
            )
            panel["sampling"]["max_tokens"] = reviewed["maximum_context"] - request["prompt_tokens"]
            panel["engine"].update(
                enable_lora=False, return_sampling_mask=False, gpu_memory_utilization=0.9
            )
            for key in LORA_ENGINE_KEYS:
                panel["engine"].pop(key, None)
            save_json(root / "config.json", panel)
            folders.append(str(root))
    save_json(out / "pending-validation.json", folders)
    return folders


def record_validation(config: dict, out: Path) -> None:
    state = read_json(out / "state.json")
    grouped = {}
    for name in read_json(out / "pending-validation.json"):
        root = Path(name)
        case = read_json(root / "config.json")["cases"][0]
        result = read_json(root / "comparison.json")
        entry = grouped.setdefault(
            case["label"],
            {"checkpoint": case["checkpoint"], "step": case["initial_step"], "projects": []},
        )
        entry["projects"].append(result["metrics"])
    for entry in grouped.values():
        entry["panel"] = aggregate_panel(entry.pop("projects"))
        state["validation_history"].append(entry)
    state["best_validation"] = select_checkpoint(
        state["validation_history"],
        minimum_improvement=scope(config)["validation_minimum_improvement"],
    )
    state["last_validation_epoch"] = utc_now().timestamp()
    save_json(out / "state.json", state)


# --- Final evaluation


def evaluation_configs(config: dict, out: Path) -> list[str]:
    """Freeze the trained checkpoint, then write one benchmark config per held-out project."""
    from transformers import AutoTokenizer

    from benchmark_contract import load_assets

    reviewed = scope(config)
    state = read_json(out / "state.json")
    effort = reviewed["evaluation_effort"]
    cases = evaluation_cases(state)
    save_json(
        out / "final-identity.json",
        {
            "cases": cases,
            "locked_utc": utc_now().isoformat(),
            "selection": "latest accepted checkpoint, frozen before test",
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(config["cycle"]["model_path"], local_files_only=True)
    projects = []
    for name in config["evaluation_projects"]:
        template = read_json(
            Path(config["runtime_host"]) / reviewed["evaluation_template"].format(project=name)
        )
        template.pop("evaluation_deadline_override", None)
        template.pop("repeat_evaluation", None)
        template["template_options"]["reasoning_effort"] = effort
        assets = load_assets(
            Path(config["evaluation_assets"]), "solidity", name, output_format="markdown"
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": assets["prompt"]}], **template["template_options"]
        )
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        run_id = f"{out.name}-evaluation-{name}"
        root = new_result_dir(config, run_id)
        template.update(
            run_id=run_id,
            cases=cases,
            generation_order=[case["label"] for case in cases],
            weekend_schedule_id=reviewed["schedule_id"],
            absolute_deadline=reviewed["final_deadline"],
            seconds_limit=reviewed["evaluation_seconds_limit"],
            judge_operation_cap_usd=reviewed["evaluation_judge_cap_usd"],
            judge_ledger=config["judge_ledgers"]["evaluation"],
            judge_cap_usd=reviewed["evaluation_judge_cap_usd"],
            prompt_token_count=len(ids),
            assets=config["evaluation_assets"],
            merged_model_path=merged_path(config, state),
            source_plan=f"/results/{out.name}/source-plan.json",
        )
        template["sampling"]["max_tokens"] = reviewed["maximum_context"] - len(ids)
        save_json(root / "config.json", template)
        projects.append(
            {
                "name": name,
                "known_gt_count": len(assets["ground_truth"]),
                "run_id": run_id,
                "run_root": f"/results/{run_id}",
            }
        )
    suite = {
        "run_id": out.name + "-evaluation",
        "runtime_host": config["runtime_host"],
        "absolute_deadline": reviewed["final_deadline"],
        "projects": projects,
        "trained_label": cases[-1]["label"],
        "sampling": {"n": 1},
        "template_options": {"reasoning_effort": effort},
        "output_format": "markdown",
        "stop_on_project_failure": False,
        "judge_ledger": config["judge_ledgers"]["evaluation"],
        "rate_usd_per_hour": reviewed["rate_usd_per_hour"],
    }
    root = new_result_dir(config, out.name + "-evaluation")
    save_json(root / "config.json", suite)
    return [str(root)]
