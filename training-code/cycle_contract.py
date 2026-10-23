"""Validated corrected-cycle configuration, complete decision and phase receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import schedules
from benchmark_free_text import SCOPE_PATH as BENCHMARK_SCOPE_PATH
from grpo.placement import docker_gpus, load_options, rollout_gpus
from runtime_io import read_json, sha256_file

SCOPE_PATH = Path(__file__).with_name("cycle_scope.json")

# A cycle config must equal the reviewed value of each of these.
PROTECTED_KEYS = (
    "model_id",
    "model_revision",
    "base_precision",
    "native_adapter_precision",
    "rollout_adapter_precision",
    "sampler_backend",
    "optimizer_kind",
    "initial_step",
    "target_step",
    "learning_rate",
    "epsilon",
    "correction_max",
    "max_grad_norm",
    "offload_layers",
    "prefetch_depth",
    "tiled_mlp",
    "update_placement",
    "seed",
    "sampling",
    "retry_limit",
    "rate_usd_per_hour",
    "absolute_deadline",
)
CONDITIONS = (
    "complete_group",
    "known_grades",
    "sampling_support",
    "source_worker_transfer",
    "finite_loss",
    "nonzero_gradient",
    "correction_health",
    "changed_weights",
    "optimizer_progress",
    "checkpoint_saved",
    "saved_adapter_identity",
    "weight_version_chain",
    "precision_declared",
    "repeat_drift_retained",
)


def evaluate_cycle(evidence: dict[str, bool]) -> dict[str, Any]:
    if set(evidence) != set(CONDITIONS) or any(
        type(value) is not bool for value in evidence.values()
    ):
        raise ValueError("Every corrected-cycle condition requires explicit boolean evidence")
    failures = [name for name in CONDITIONS if not evidence[name]]
    return {"approved": not failures, "failed_conditions": failures, "scope": "one_corrected_cycle"}


def validate_config(config: dict[str, Any]) -> None:
    """One update within cycle_scope.json and the reviewed schedule the config names."""
    reviewed = schedules.scope(config)
    limits = reviewed["training"]
    context = reviewed["maximum_context"]
    prompt_count = config.get("prompt_token_count")
    if type(prompt_count) is not int or not 0 < prompt_count < context:
        raise ValueError("Measured prompt token count required")
    initial, target = config.get("initial_step"), config.get("target_step")
    if (
        type(initial) is not int
        or type(target) is not int
        or target != initial + 1
        or not limits["initial_step"] <= initial < target <= limits["maximum_step"]
        or not config.get("row_id")
        or not config.get("dataset_workspace")
    ):
        raise ValueError("Cycle step or row differs from the reviewed schedule")
    if config.get("policy_reasoning_effort") != limits["policy_reasoning_effort"]:
        raise ValueError("Policy reasoning effort differs from reviewed scope")
    required = read_json(SCOPE_PATH)
    required["engine"].update(
        max_model_len=context, gpu_memory_utilization=limits["gpu_memory_utilization"]
    )
    # Absent means vLLM's single-GPU default, the only setting older schedules ran.
    if "tensor_parallel_size" in limits:
        required["engine"]["tensor_parallel_size"] = limits["tensor_parallel_size"]
    required["sampling"]["max_tokens"] = context - prompt_count
    required.update(
        initial_step=initial,
        target_step=target,
        offload_layers=limits["offload_layers"],
        prefetch_depth=limits["prefetch_depth"],
        tiled_mlp=limits["tiled_mlp"],
        learning_rate=limits["learning_rate"],
        update_placement=limits.get("update_placement", required["update_placement"]),
        absolute_deadline=reviewed["training_cutoff"],
    )
    if config.get("engine") != required["engine"]:
        raise ValueError(
            "Cycle engine differs from reviewed scope: "
            + differing(config.get("engine"), required["engine"])
        )
    changed = [name for name in PROTECTED_KEYS if config.get(name) != required[name]]
    if changed:
        raise ValueError(
            "Model, objective, precision or schedule differs from reviewed scope: "
            + ", ".join(
                differing(config.get(name), required[name], name)
                if isinstance(required[name], dict)
                else f"{name}={config.get(name)!r} (reviewed {required[name]!r})"
                for name in changed
            )
        )
    load_options(config["update_placement"])
    rollout_gpus(config["engine"])
    for name in ("seconds_limit", "judge_seconds_limit"):
        if type(config.get(name)) is not int or not 0 < config[name] <= limits[name]:
            raise ValueError(f"Invalid reviewed time budget: {name}")
    bounds = {
        **required["correction_bounds"],
        "judge_operation_cap_usd": [0, limits["judge_operation_cap_usd"]],
    }
    for name, (lower, upper) in bounds.items():
        value = config.get(name)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise ValueError(f"Invalid reviewed correction or spending bound: {name}")
    if (
        config["judge_operation_cap_usd"] == 0
        or type(config.get("projection_chunk")) is not int
        or config["projection_chunk"] < 1
    ):
        raise ValueError("Positive operation budget and projection chunk required")


def differing(actual: Any, reviewed: dict[str, Any], prefix: str = "") -> str:
    """Name each field that differs, so a rejected config says what to change."""
    actual = actual if isinstance(actual, dict) else {}
    label = f"{prefix}." if prefix else ""
    return ", ".join(
        f"{label}{key}={actual.get(key)!r} (reviewed {reviewed.get(key)!r})"
        for key in sorted(set(actual) | set(reviewed))
        if actual.get(key) != reviewed.get(key)
    )


def prepare_cycle(config: dict[str, Any], out: Path) -> None:
    validate_config(config)
    if (out / "prepared.json").exists() or any(
        (out / name).exists() for name in ("rollout", "update", "grading")
    ):
        raise FileExistsError("GPU-cycle output must be fresh")
    deadline = datetime.fromisoformat(config["absolute_deadline"])
    if (deadline - datetime.now(UTC)).total_seconds() < config["seconds_limit"]:
        raise ValueError("Insufficient remaining machine budget")
    checkpoint = Path(config["checkpoint"])
    for name in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    digest = hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
    if digest != config["initial_adapter_sha256"]:
        raise ValueError("Accepted source checkpoint identity differs")
    if (
        json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
        != config["initial_step"]
    ):
        raise ValueError("Accepted source checkpoint step differs")
    for name in ("source_plan", "request", "prior_analysis"):
        if not Path(config[name]).is_file():
            raise FileNotFoundError(config[name])
    if not Path(config["model_path"]).is_dir():
        raise FileNotFoundError(config["model_path"])
    (out / "prepared.json").write_text(json.dumps(config, indent=2) + "\n")
    plan = json.loads(Path(config["source_plan"]).read_text())
    plan["judge_operation"] = {"id": config["run_id"], "cap_usd": config["judge_operation_cap_usd"]}
    if config.get("judge_env_file"):
        plan["judge_env_file"] = config["judge_env_file"]
    plan["judge_retry_limit"] = config["retry_limit"]
    plan["dataset_workspace"] = config["dataset_workspace"]
    plan["selected_row_id"] = config["row_id"]
    # The judge and the reward come from the reviewed files, never from the cycle config.
    plan["reward"] = schedules.scope(config)["training"]["reward"]
    plan["benchmark_judge"] = read_json(BENCHMARK_SCOPE_PATH)["benchmark_judge"]
    (out / "judge-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    write_event(
        out, "cycle_start", initial_step=config["initial_step"], target_step=config["target_step"]
    )


def validate_policy_profile(config: dict, request: dict, plan: dict) -> None:
    effort = config.get("policy_reasoning_effort")
    if effort is None:
        return
    options = request.get("template_options", {})
    if (
        options.get("enable_thinking") is not True
        or options.get("reasoning_effort") != effort
        or plan.get("sampling_profile", {}).get("policy_reasoning_effort") != effort
    ):
        raise ValueError(
            "Actual request/plan reasoning profile differs from training configuration"
        )
    if request["full_completion_ceiling"] != config["sampling"]["max_tokens"]:
        raise ValueError("Request completion ceiling differs from training configuration")


def finish_cycle(config: dict[str, Any], out: Path) -> dict[str, Any]:
    source = json.loads((out / "rollout/receipt.json").read_text())
    update = json.loads((out / "update/receipt.json").read_text())
    samples = json.loads((out / "rollout/samples.json").read_text())
    grades = json.loads((out / "grading/rewards.json").read_text())
    prior = json.loads(Path(config["prior_analysis"]).read_text())
    request = read_json(config["request"])
    checkpoint = out / f"update/checkpoint-{config['target_step']}"
    evidence = {
        "complete_group": len(samples) == 4
        and all(
            sample.get("row_id") == config["row_id"] == request["row_id"]
            and sample.get("prompt_ids") == request["prompt_ids"]
            for sample in samples
        ),
        "known_grades": len(grades) == 4
        and all(type(value) in (int, float) and math.isfinite(value) for value in grades),
        "sampling_support": all(
            sample.get("sampling_support_ids") is not None for sample in samples
        ),
        "source_worker_transfer": source["transfer"]["cast_transfer_verified"],
        "finite_loss": math.isfinite(update["loss"]),
        "nonzero_gradient": math.isfinite(update["gradient_norm"]) and update["gradient_norm"] > 0,
        "correction_health": update["correction_clip_fraction"] <= config["max_clip_fraction"]
        and update["correction_ess_fraction"] >= config["min_ess_fraction"],
        "changed_weights": update["adapter_weights_changed"],
        "optimizer_progress": update["optimizer_steps"] == [config["target_step"]],
        "checkpoint_saved": all(
            (checkpoint / name).is_file()
            for name in (
                "adapter_model.safetensors",
                "optimizer.pt",
                "rng_state.pth",
                "scheduler.pt",
                "trainer_state.json",
            )
        ),
        # vLLM's copy of this adapter is verified when the next visit's rollout loads it.
        "saved_adapter_identity": (checkpoint / "adapter_model.safetensors").is_file()
        and sha256_file(checkpoint / "adapter_model.safetensors") == update["adapter_sha256"]
        and update["weight_version"] == f"step-{config['target_step']}:{update['adapter_sha256']}",
        "weight_version_chain": update["weight_version"] != source["weight_version"]
        and update["source_weight_version"] == source["weight_version"]
        and all(sample["weight_version"] == source["weight_version"] for sample in samples),
        "precision_declared": source["native_adapter_precision"] == "float32"
        and source["rollout_adapter_precision"] == "bfloat16",
        "repeat_drift_retained": "repeat" in prior
        and all(
            math.isfinite(prior["repeat"].get(name, float("nan")))
            for name in ("raw_max_abs_logprob_difference", "processed_max_abs_logprob_difference")
        ),
    }
    result = {
        "decision": evaluate_cycle(evidence),
        "evidence": evidence,
        "historical_exact_parity_approved": False,
        "prior_repeat": prior["repeat"],
        "optimizer_updates": 1,
        "initial_step": config["initial_step"],
        "target_step": config["target_step"],
        "quality_gain_established": False,
    }
    (out / "cycle-result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_event(out, "cycle_end", approved=result["decision"]["approved"])
    return result


def write_event(out: Path, event: str, **details: Any) -> None:
    value = {
        "phase": out.name,
        "event": event,
        "utc": datetime.now(UTC).isoformat(),
        **details,
    }
    with (out / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("settings", "prepare", "finish"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.phase == "settings":
        print("\n".join(launcher_settings(config)))
        return 0
    if args.out is None:
        parser.error(f"{args.phase} requires --out")
    if args.phase == "prepare":
        prepare_cycle(config, args.out)
        return 0
    return 0 if finish_cycle(config, args.out)["decision"]["approved"] else 1


def launcher_settings(config: dict[str, Any]) -> list[str]:
    """Lines read by launch-cycle.sh: images, capped time limits, skip policy, phase GPUs."""
    until_deadline = int(
        (datetime.fromisoformat(config["absolute_deadline"]) - datetime.now(UTC)).total_seconds()
    )
    return [
        config["native_image"],
        config["rollout_image"],
        str(min(config["seconds_limit"], until_deadline)),
        str(config["judge_seconds_limit"]),
        str(config.get("skip_uninformative_groups", False)).lower(),
        docker_gpus(config["update_placement"]["gpus"]),
        docker_gpus(rollout_gpus(config["engine"])),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
