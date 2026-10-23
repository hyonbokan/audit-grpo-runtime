"""Write a training run's config, source plan and judge ledgers from its reviewed schedule.

Runs in the CPU image (launch-segment.sh), where runs/ is /results and the repository /runtime.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import initial_checkpoint
import schedules
from benchmark_free_text import SCOPE_PATH as BENCHMARK_SCOPE_PATH
from cycle_contract import SCOPE_PATH as CYCLE_SCOPE_PATH
from runtime_io import read_json, save_json, sha256_file

RESULTS = Path("/results")
LOCK = Path("/runtime/images/gpu-requirements-cu129.lock")
# The update refuses to load the model unless these installed versions equal the plan's.
PINNED_PACKAGES = (
    "unsloth",
    "unsloth_zoo",
    "trl",
    "transformers",
    "peft",
    "torch",
    "datasets",
    "accelerate",
)
# The model's own context; the update fits were measured with it. The schedule caps generation.
MODEL_CONTEXT_LIMIT = 262144
# The projection chunk every accepted cycle used.
PROJECTION_CHUNK = 32
# Four equal rewards carry no advantage, so such a group gives no update.
SKIP_UNINFORMATIVE_GROUPS = True
LEDGER_NAMES = {
    "training": "judge-training.json",
    "validation": "judge-validation.json",
    "evaluation": "judge-evaluation.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--runtime-host", required=True)
    parser.add_argument("--artifact-host", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--judge-env-file", required=True)
    args = parser.parse_args()
    host = {
        "runtime_host": args.runtime_host,
        "artifact_host": args.artifact_host,
        "hf_repo": args.hf_repo,
        "judge_env_file": args.judge_env_file,
    }
    print(write_run(args.schedule, results=RESULTS, lock=LOCK, host=host))
    return 0


def write_run(schedule_id: str, *, results: Path, lock: Path, host: dict[str, str]) -> Path:
    """The run directory's config.json; the directory must exist and be empty."""
    reviewed = schedules.schedule(schedule_id)
    if "launch" not in reviewed:
        raise ValueError(f"Schedule {schedule_id!r} names no launch inputs; it cannot start a run")
    launch = reviewed["launch"]
    run = results / schedule_id
    if not run.is_dir() or any(run.iterdir()):
        raise FileExistsError(f"{run} must be an empty run directory made by the launcher")
    corpus = results / launch["corpus"] / "manifest.json"
    if sha256_file(corpus) != launch["corpus_manifest_sha256"]:
        raise ValueError("Corpus manifest differs from the reviewed schedule")
    benchmark = results / launch["benchmark_assets"] / "BENCHMARK_MANIFEST.json"
    if sha256_file(benchmark) != launch["benchmark_manifest_sha256"]:
        raise ValueError("Benchmark manifest differs from the reviewed schedule")
    start, adapter_sha256 = start_checkpoint(reviewed, results)
    ledgers = campaign_ledgers(reviewed, results)
    save_json(run / "source-plan.json", source_plan(reviewed, lock, ledgers["training"]))
    shutil.copyfile(results / launch["prior_analysis"], run / "prior-analysis.json")
    cycle = cycle_config(reviewed, start, adapter_sha256, host)
    runtime = host["runtime_host"]
    config = {
        "run_id": schedule_id,
        "schedule_id": schedule_id,
        "training_cutoff": reviewed["training_cutoff"],
        "final_deadline": reviewed["final_deadline"],
        "cycle": cycle,
        "initial_checkpoint": start,
        "initial_adapter_sha256": adapter_sha256,
        "dataset_host": f"{runtime}/runs/{launch['corpus']}",
        "dataset_container": container(launch["corpus"]),
        "dataset_manifest_sha256": launch["corpus_manifest_sha256"],
        "benchmark_assets": f"{runtime}/runs/{launch['benchmark_assets']}",
        "benchmark_manifest_sha256": launch["benchmark_manifest_sha256"],
        "runtime_host": runtime,
        "run_host": f"{runtime}/runs/{schedule_id}",
        "artifact_host": host["artifact_host"],
        "epochs": 1,
        "seed": cycle["seed"],
        "hf_repo": host["hf_repo"],
        "judge_env_file": host["judge_env_file"],
        "backup_container": container(f"{schedule_id}-backups"),
        "judge_ledger": ledgers["training"],
        "judge_ledgers": ledgers,
        "timezone": "UTC",
    }
    save_json(run / "config.json", config)
    return run / "config.json"


# --- Inputs


def start_checkpoint(reviewed: dict[str, Any], results: Path) -> tuple[str, str]:
    """The starting checkpoint's container path and adapter hash; a fresh start makes step 0."""
    start = reviewed["launch"]["start"]
    training = reviewed["training"]
    if set(start) == {"checkpoint"}:
        relative = start["checkpoint"]
    elif set(start) == {"template", "seed"}:
        relative = f"{reviewed['schedule_id']}-start/checkpoint-0"
        expected = {
            "kind": "initial",
            "initial_step": 0,
            "seed": start["seed"],
            "learning_rate": training["learning_rate"],
        }
        out = results / relative
        if not out.exists():
            initial_checkpoint.create(
                results / start["template"], out, start["seed"], training["learning_rate"]
            )
        elif read_json(out / "continuation_schedule.json") != expected:
            raise ValueError(f"{out} was made with another seed or learning rate")
    else:
        raise ValueError(f"launch.start must be {{checkpoint}} or {{template, seed}}: {start}")
    checkpoint = results / relative
    step = read_json(checkpoint / "trainer_state.json")["global_step"]
    if step != training["initial_step"]:
        raise ValueError(f"Start checkpoint is at step {step}, not {training['initial_step']}")
    return container(relative), sha256_file(checkpoint / "adapter_model.safetensors")


def campaign_ledgers(reviewed: dict[str, Any], results: Path) -> dict[str, str]:
    """Shared judge ledgers for every segment of the campaign, made once with the reviewed caps."""
    campaign = reviewed["launch"]["campaign"]
    caps = {
        "training": reviewed["judge_cap_usd"],
        "validation": reviewed["evaluation_judge_cap_usd"],
        "evaluation": reviewed["evaluation_judge_cap_usd"],
    }
    paths = {}
    for label, name in LEDGER_NAMES.items():
        path = results / campaign / name
        if not path.exists():
            save_json(path, {"cap_usd": caps[label], "calls": []})
        elif read_json(path)["cap_usd"] != caps[label]:
            raise ValueError(f"{path} has another cap than the reviewed {caps[label]}")
        paths[label] = container(f"{campaign}/{name}")
    return paths


def source_plan(reviewed: dict[str, Any], lock: Path, training_ledger: str) -> dict[str, Any]:
    """Only what the rollout, judge worker and update read from the plan."""
    scope = read_json(CYCLE_SCOPE_PATH)
    return {
        "version": reviewed["schedule_id"],
        "model": {
            "id": scope["model_id"],
            "revision": scope["model_revision"],
            "context_limit": MODEL_CONTEXT_LIMIT,
        },
        "package_versions": pinned_versions(lock),
        "sampling_profile": {
            "id": reviewed["schedule_id"],
            "policy_reasoning_effort": reviewed["training"]["policy_reasoning_effort"],
        },
        "access": {
            "judge_usd_cap": reviewed["judge_cap_usd"],
            "local_judge_budget": {"path": training_ledger},
        },
        "dataset_workspace": container(reviewed["launch"]["corpus"]),
    }


def pinned_versions(lock: Path) -> dict[str, str]:
    found = {
        match[1].lower().replace("-", "_"): match[2]
        for match in re.finditer(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", lock.read_text(), re.M)
    }
    missing = [name for name in PINNED_PACKAGES if name not in found]
    if missing:
        raise ValueError(f"{lock} pins none of {missing}")
    return {name: found[name] for name in PINNED_PACKAGES}


# --- Cycle


def cycle_config(
    reviewed: dict[str, Any], start: str, adapter_sha256: str, host: dict[str, str]
) -> dict[str, Any]:
    """The cycle every visit copies; controller.next_cycle adds the row, request and steps."""
    scope = read_json(CYCLE_SCOPE_PATH)
    images = read_json(BENCHMARK_SCOPE_PATH)
    training = reviewed["training"]
    run = reviewed["schedule_id"]
    engine = dict(
        scope["engine"],
        max_model_len=reviewed["maximum_context"],
        gpu_memory_utilization=training["gpu_memory_utilization"],
    )
    # Absent means vLLM's single-GPU default, as in cycle_contract.
    if "tensor_parallel_size" in training:
        engine["tensor_parallel_size"] = training["tensor_parallel_size"]
    reviewed_keys = (
        "model_id",
        "model_revision",
        "base_precision",
        "native_adapter_precision",
        "rollout_adapter_precision",
        "sampler_backend",
        "optimizer_kind",
        "epsilon",
        "correction_max",
        "max_grad_norm",
        "seed",
        "retry_limit",
        "rate_usd_per_hour",
    )
    model_cache = "models--" + scope["model_id"].replace("/", "--")
    return {
        "run_id": run,
        "schedule_id": run,
        **{name: scope[name] for name in reviewed_keys},
        "native_image": images["native_image"],
        "rollout_image": images["rollout_image"],
        "model_path": (
            f"/workspace/cache/huggingface/hub/{model_cache}/snapshots/{scope['model_revision']}"
        ),
        "checkpoint": start,
        "initial_adapter_sha256": adapter_sha256,
        "source_plan": container(f"{run}/source-plan.json"),
        "prior_analysis": container(f"{run}/prior-analysis.json"),
        "initial_step": training["initial_step"],
        "target_step": training["initial_step"] + 1,
        "learning_rate": training["learning_rate"],
        "max_clip_fraction": scope["correction_bounds"]["max_clip_fraction"][1],
        "min_ess_fraction": scope["correction_bounds"]["min_ess_fraction"][0],
        "projection_chunk": PROJECTION_CHUNK,
        "offload_layers": training["offload_layers"],
        "prefetch_depth": training["prefetch_depth"],
        "tiled_mlp": training["tiled_mlp"],
        # Absent means the reviewed single-GPU placement, as in cycle_contract.
        "update_placement": training.get("update_placement", scope["update_placement"]),
        "sampling": dict(scope["sampling"]),
        "engine": engine,
        "seconds_limit": training["seconds_limit"],
        "judge_seconds_limit": training["judge_seconds_limit"],
        "judge_operation_cap_usd": training["judge_operation_cap_usd"],
        "absolute_deadline": reviewed["training_cutoff"],
        "dataset_workspace": container(reviewed["launch"]["corpus"]),
        "judge_env_file": host["judge_env_file"],
        "policy_reasoning_effort": training["policy_reasoning_effort"],
        "skip_uninformative_groups": SKIP_UNINFORMATIVE_GROUPS,
    }


def container(relative: str) -> str:
    """A runs/ path as every container sees it."""
    return f"/results/{relative}"


if __name__ == "__main__":
    raise SystemExit(main())
