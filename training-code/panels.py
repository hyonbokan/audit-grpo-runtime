"""The locked final evaluation suite and the merge identity it evaluates."""

from __future__ import annotations

from pathlib import Path

from benchmark_free_text import evaluation_row_ids
from runtime_io import new_result_dir, read_json, save_json, sha256_file, utc_now
from schedules import scope


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


# --- Final evaluation


def evaluation_projects() -> list[str]:
    """Every final evaluation runs the whole benchmark: the five Solidity repositories."""
    return [row_id.split("/", 1)[1] for row_id in evaluation_row_ids()]


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
    for name in evaluation_projects():
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
