"""A fixed four-draw project from the validation or clean final-test panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from generation import validate_request
from runtime_io import save_json as save


def validate_panel_config(config: dict[str, Any]) -> None:
    scope = json.loads(Path(__file__).parents[1].joinpath("cycle_scope.json").read_text())
    if config.get("policy_reasoning_effort") == "xhigh":
        scope["sampling"] = scope["audited_xhigh"]["sampling"]
    weekend = bool(config.get("weekend_schedule_id"))
    if weekend:
        import schedules

        reviewed = schedules.scope(config)
        scope["absolute_deadline"] = reviewed["training_cutoff"]
        scope["sampling"]["max_tokens"] = reviewed["maximum_context"] - config["prompt_token_count"]
        scope["engine"]["max_model_len"] = reviewed["maximum_context"]
        scope["full_training"]["maximum_context"] = reviewed["maximum_context"]
        if config.get("merged_model_path"):
            scope["engine"].update(enable_lora=False, gpu_memory_utilization=0.9)
            for key in ("lora_dtype", "max_loras", "max_lora_rank"):
                scope["engine"].pop(key)
    for key in (
        "model_id",
        "model_revision",
        "base_precision",
        "native_adapter_precision",
        "rollout_adapter_precision",
        "sampler_backend",
        "sampling",
        "retry_limit",
        "absolute_deadline",
    ):
        if config.get(key) != scope[key]:
            raise ValueError(f"Panel differs from fixed inference contract: {key}")
    if config.get("split") not in ("calibration", "evaluation") or not config.get(
        "dataset_workspace"
    ):
        raise ValueError("Validation or untouched test split required")
    engine = config["engine"]
    if (
        not scope["engine"]["max_model_len"]
        <= engine["max_model_len"]
        <= scope["full_training"]["maximum_context"]
    ):
        raise ValueError("Panel context outside reviewed range")
    if {
        **engine,
        "max_model_len": scope["engine"]["max_model_len"],
        "return_sampling_mask": True,
    } != scope["engine"] or engine["return_sampling_mask"]:
        raise ValueError("Panel requires fixed engine without training masks")
    for name, ceiling in [
        ("seconds_limit", 21600 if weekend else 5400),
        ("judge_seconds_limit", 1800),
        ("judge_operation_cap_usd", 20 if weekend else 150),
    ]:
        if (
            type(config[name]) not in (int, float)
            or not math.isfinite(config[name])
            or not 0 < config[name] <= ceiling
        ):
            raise ValueError("Invalid panel time/spending bound")
    if len(config["cases"]) != 1:
        raise ValueError("One project/checkpoint arm per panel invocation required")
    case = config["cases"][0]
    if case["label"] != "base" and (
        type(case["initial_step"]) is not int
        or not 2 <= case["initial_step"] <= (512 if weekend else 128)
    ):
        raise ValueError("Panel checkpoint outside approved training scope")


def project_metrics(row_id, gt_ids, grades, samples):
    if not gt_ids or len(set(gt_ids)) != len(gt_ids) or len(grades) != 4 or len(samples) != 4:
        raise ValueError("Four draws and a nonempty unique known-GT denominator required")
    if any(
        type(g["reward"]) not in (int, float)
        or not math.isfinite(g["reward"])
        or g["components"]["withheld_reasons"]
        for g in grades
    ):
        raise ValueError("Unknown grades cannot enter checkpoint selection")
    exact, partial = [], []
    for grade in grades:
        hits = {p[0] for p in grade["components"].get("unambiguous_pairs", [])}
        roots = set(hits)
        for response in grade.get("judge_responses", []):
            roots.update(
                a["gt_id"]
                for a in response.get("assessments", [])
                if a.get("same_defect_operation")
            )
        if not hits.issubset(gt_ids) or not roots.issubset(gt_ids):
            raise ValueError("Unknown GT cannot inflate panel coverage")
        exact.append(len(hits) / len(gt_ids))
        partial.append(len(roots) / len(gt_ids))
    # Canonical schema validity before fence removal is a separate metric.
    from core.schemas.llm_schema import LLMFindingList

    strict = []
    for sample in samples:
        try:
            LLMFindingList.model_validate_json(sample["final_text"])
            valid = sample["finish_reason"] == "stop"
        except ValueError:
            valid = False
        strict.append(valid)
    return {
        "row_id": row_id,
        "eligible_gt_ids": sorted(gt_ids),
        "draws": 4,
        "unknown_draws": 0,
        "exact_coverage": sum(exact) / 4,
        "partial_coverage": sum(partial) / 4,
        "format_validity": sum(strict) / 4,
        "normalized_format_validity": sum(g["components"]["format_valid"] for g in grades) / 4,
        "per_draw_exact_coverage": exact,
        "per_draw_partial_coverage": partial,
        "truncated_draws": sum(s["finish_reason"] != "stop" for s in samples),
        "mean_findings": sum(g["components"]["raw_findings"] for g in grades) / 4,
    }


def private_row(config):
    dataset = Path(config["dataset_workspace"])
    rows = [
        json.loads(line)
        for line in (dataset / f"data/private/{config['split']}.jsonl").read_text().splitlines()
    ]
    selected = [r for r in rows if r["row_id"] == config["row_id"]]
    if len(selected) != 1:
        raise ValueError("Selected project is not in the frozen held-out partition")
    row = selected[0]
    gt = sorted(t["id"] for t in row["scorer_metadata"]["targets"] if not t["required_context"])
    if not gt:
        raise ValueError("No eligible held-out targets")
    return row, gt


def validate_grading_join(samples, graded_samples, metrics, row_id, gt_ids):
    if (
        samples != graded_samples
        or metrics["row_id"] != row_id
        or metrics["eligible_gt_ids"] != sorted(gt_ids)
    ):
        raise ValueError("Grading does not bind the exact samples and known-GT denominator")


def prepare(config, out):
    validate_panel_config(config)
    if (out / "prepared.json").exists():
        raise FileExistsError("Panel output must be fresh")
    if (
        datetime.fromisoformat(config["absolute_deadline"]) - datetime.now(UTC)
    ).total_seconds() < config["seconds_limit"]:
        raise ValueError("Insufficient remaining machine budget")
    row, gt = private_row(config)
    policy = [
        json.loads(line)
        for line in (Path(config["dataset_workspace"]) / f"data/policy/{config['split']}.jsonl")
        .read_text()
        .splitlines()
    ]
    messages = [r for r in policy if r["row_id"] == config["row_id"]]
    if len(messages) != 1 or set(messages[0]) != {"row_id", "messages"}:
        raise ValueError("Private/policy project join differs")
    request = json.loads(Path(config["request"]).read_text())
    validate_request(request)
    if (
        request["row_id"] != config["row_id"]
        or request["max_model_len"] != config["engine"]["max_model_len"]
    ):
        raise ValueError("Frozen complete prompt request differs")
    if (
        request["model"]["id"] != config["model_id"]
        or request["model"]["revision"] != config["model_revision"]
    ):
        raise ValueError("Frozen model request differs")
    if request["full_completion_ceiling"] != config["sampling"]["max_tokens"]:
        raise ValueError("Full completion ceiling differs from fixed panel budget")
    case = config["cases"][0]
    if case["label"] != "base":
        checkpoint = Path(case["checkpoint"])
        if (
            hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
            != case["initial_adapter_sha256"]
        ):
            raise ValueError("Checkpoint hash differs")
        if (
            json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
            != case["initial_step"]
        ):
            raise ValueError("Checkpoint step differs")
    save(out / "request.json", request)
    save(
        out / "private-identity.json",
        {
            "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
            "eligible_gt_ids": gt,
            "request_sha256": hashlib.sha256((out / "request.json").read_bytes()).hexdigest(),
        },
    )
    plan = json.loads(Path(config["source_plan"]).read_text())
    plan.update(
        judge_operation={"id": config["run_id"], "cap_usd": config["judge_operation_cap_usd"]},
        dataset_workspace=config["dataset_workspace"],
        selected_row_id=config["row_id"],
        judge_retry_limit=0,
        judge_env_file=config["judge_env_file"],
    )
    save(out / "judge-plan.json", plan)
    save(
        out / f"{case['label']}-config.json",
        {**config, **case, "request": str(out / "request.json")},
    )
    save(out / "prepared.json", config)


def finish(config, out):
    validate_panel_config(config)
    if json.loads((out / "prepared.json").read_text()) != config:
        raise ValueError("Panel configuration changed")
    row, gt = private_row(config)
    identity = json.loads((out / "private-identity.json").read_text())
    if (
        identity["row_sha256"]
        != hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        or identity["eligible_gt_ids"] != gt
    ):
        raise ValueError("Held-out private denominator changed")
    if (
        identity["request_sha256"]
        != hashlib.sha256((out / "request.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Frozen prompt request changed")
    request = json.loads((out / "request.json").read_text())
    case = config["cases"][0]
    label = case["label"]
    samples = json.loads((out / label / "samples.json").read_text())
    version = (
        f"base:{config['model_revision']}"
        if label == "base"
        else f"step-{case['initial_step']}:{case['initial_adapter_sha256']}"
    )
    if len(samples) != 4 or any(
        s["row_id"] != config["row_id"]
        or s["prompt_ids"] != request["prompt_ids"]
        or s["weight_version"] != version
        or s["logprob_semantics"] != "sampling_processed"
        or len(s["completion_ids"]) != len(s["logprobs"])
        for s in samples
    ):
        raise ValueError("Panel samples differ from matched model/prompt/sampling contract")
    grades = json.loads((out / f"{label}-grading/attempt-1/grades.json").read_text())
    validate_grading_join(
        samples,
        json.loads((out / f"{label}-grading/samples.json").read_text()),
        json.loads((out / f"{label}-grading/metrics.json").read_text()),
        config["row_id"],
        gt,
    )
    metrics = project_metrics(config["row_id"], gt, grades, samples)
    result = {
        "status": "fixed_project_panel_completed",
        "checkpoint": case,
        "metrics": metrics,
        "optimizer_steps": 0,
        "samples_sha256": hashlib.sha256((out / label / "samples.json").read_bytes()).hexdigest(),
        "private_identity": identity,
        "quality_gain_established": False,
    }
    save(out / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "finish"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    (prepare if args.phase == "prepare" else finish)(config, args.out)


if __name__ == "__main__":
    main()
