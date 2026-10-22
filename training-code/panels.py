"""The locked final evaluation suite and the merge identity it evaluates."""

from __future__ import annotations

from pathlib import Path

from benchmark_free_text import evaluation_row_ids
from runtime_io import new_result_dir, read_json, save_json, sha256_file, utc_now
from schedules import scope


def trained_case(state: dict) -> dict:
    """The latest accepted checkpoint, the only arm a final evaluation generates."""
    return {
        "label": f"checkpoint-{state['current_step']}",
        "initial_step": state["current_step"],
        "checkpoint": state["current_checkpoint"],
        "initial_adapter_sha256": state["current_adapter_sha256"],
    }


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
    """Freeze the trained checkpoint, then write its config per project against the frozen base.

    Each project's config is its frozen base run's own, so the arms differ only in the weights.
    """
    from benchmark_evaluation import validate_benchmark_config

    reviewed = scope(config)
    state = read_json(out / "state.json")
    case = trained_case(state)
    results = Path(config["runtime_host"]) / "results"
    references = {}
    for name in evaluation_projects():
        base_run = reviewed["evaluation_base_reference"].format(project=name)
        root = results / base_run
        comparison = root / "comparison.json"
        if not comparison.is_file():
            raise FileNotFoundError(f"Frozen base run has no {comparison}")
        references[name] = (base_run, read_json(root / "prepared.json"), sha256_file(comparison))
    # The suite report states one draw count and effort for every project.
    shared = {
        (frozen["sampling"]["n"], frozen["template_options"]["reasoning_effort"])
        for _, frozen, _ in references.values()
    }
    if len(shared) != 1:
        raise ValueError(f"Frozen base runs differ in draws per arm or reasoning effort: {shared}")
    [(draws, effort)] = shared
    save_json(
        out / "final-identity.json",
        {
            "case": case,
            "locked_utc": utc_now().isoformat(),
            "selection": "latest accepted checkpoint, frozen before test",
        },
    )
    projects = []
    for name, (base_run, frozen, comparison_sha256) in references.items():
        run_id = f"{out.name}-evaluation-{name}"
        child = dict(
            frozen,
            run_id=run_id,
            cases=[case],
            generation_order=[case["label"]],
            merged_model_path=merged_path(config, state),
            base_reference={
                "run_root": f"/results/{base_run}",
                "comparison_sha256": comparison_sha256,
            },
            absolute_deadline=reviewed["final_deadline"],
            seconds_limit=min(frozen["seconds_limit"], reviewed["evaluation_seconds_limit"]),
            judge_operation_cap_usd=reviewed["evaluation_judge_cap_usd"],
            judge_ledger=config["judge_ledgers"]["evaluation"],
            judge_cap_usd=reviewed["evaluation_judge_cap_usd"],
        )
        validate_benchmark_config(child)
        root = new_result_dir(config, run_id)
        save_json(root / "config.json", child)
        projects.append(
            {
                "name": name,
                "known_gt_count": len(read_json(results / base_run / "private-truth.json")),
                "run_id": run_id,
                "run_root": f"/results/{run_id}",
                "base_run_root": f"/results/{base_run}",
            }
        )
    suite = {
        "run_id": out.name + "-evaluation",
        "runtime_host": config["runtime_host"],
        "absolute_deadline": reviewed["final_deadline"],
        "projects": projects,
        "trained_label": case["label"],
        "sampling": {"n": draws},
        "template_options": {"reasoning_effort": effort},
        "output_format": "free_text",
        "stop_on_project_failure": False,
        "judge_ledger": config["judge_ledgers"]["evaluation"],
        "rate_usd_per_hour": reviewed["rate_usd_per_hour"],
    }
    root = new_result_dir(config, out.name + "-evaluation")
    save_json(root / "config.json", suite)
    return [str(root)]
