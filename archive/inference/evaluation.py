"""Prepare and summarize one matched held-out checkpoint comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from generation import validate_request
from grpo.measurement import compare
from grpo.policy_prompt import load_policy_rows, policy_template_options, prompt_token_ids
from grpo.samples import write_new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("settings", "prepare", "finish"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.phase == "settings":
        print("\n".join(launcher_settings(config)))
        return 0
    if args.out is None:
        parser.error(f"{args.phase} requires --out")
    if args.phase == "prepare":
        prepare_evaluation(config, args.out)
    else:
        finish_evaluation(config, args.out)
    return 0


EVALUATION_MODULES = {"native_solidity_benchmark": "benchmark_evaluation"}


def launcher_settings(config: dict[str, Any]) -> list[str]:
    """Lines read by launch-evaluation.sh, after validating the config for its kind."""
    validate_evaluation_config(config)
    # Configs without a kind predate it and are the matched-checkpoint comparison here.
    labels = config.get("generation_order", [case["label"] for case in config["cases"]])
    conditional = config.get("continuation_reference") or config.get("content_panel")
    return [
        config["native_image"],
        config["rollout_image"],
        str(config["seconds_limit"]),
        str(config["judge_seconds_limit"]),
        EVALUATION_MODULES.get(config.get("evaluation_kind"), "evaluation"),
        " ".join(labels),
        "conditional" if conditional else "strict",
        config.get("split", "evaluation"),
        "overlap" if config.get("overlap_grading") else "serial",
    ]


def validate_evaluation_config(config: dict[str, Any]) -> None:
    kind = config.get("evaluation_kind")
    if kind is not None and kind not in EVALUATION_MODULES:
        raise ValueError(f"Unknown evaluation_kind {kind!r}; known: {sorted(EVALUATION_MODULES)}")
    if config.get("evaluation_kind") == "native_solidity_benchmark":
        from benchmark_evaluation import validate_benchmark_config

        validate_benchmark_config(config)
        return
    reviewed = json.loads(Path(__file__).parents[1].joinpath("evaluation_scope.json").read_text())
    paths = {"workspace", "source_plan", "model_path"}
    bounds = {"seconds_limit", "judge_seconds_limit", "judge_operation_cap_usd"}
    for name, value in reviewed.items():
        if name not in paths | bounds | {"cases"} and config.get(name) != value:
            raise ValueError(f"Evaluation differs from reviewed contract: {name}")
    for name in bounds:
        if type(config.get(name)) not in (int, float) or not 0 < config[name] <= reviewed[name]:
            raise ValueError(f"Invalid evaluation budget: {name}")
    if "initial_step" in config and not any(
        all(
            config.get(key) == case.get(key)
            for key in ("initial_step", "checkpoint", "initial_adapter_sha256")
        )
        for case in config["cases"]
    ):
        raise ValueError("Evaluation checkpoint is outside the declared comparison")


def prepare_evaluation(config: dict[str, Any], out: Path, *, tokenizer: Any = None) -> None:
    validate_evaluation_config(config)
    if (out / "prepared.json").exists():
        raise FileExistsError("Evaluation output must be fresh")
    deadline = datetime.fromisoformat(config["absolute_deadline"])
    if (deadline - datetime.now(UTC)).total_seconds() < config["seconds_limit"]:
        raise ValueError("Insufficient remaining machine budget")
    rows = load_policy_rows(Path(config["workspace"]))["evaluation"]
    selected = [row for row in rows if row["row_id"] == config["row_id"]]
    if len(selected) != 1:
        raise ValueError("One existing held-out policy scope required")
    plan = json.loads(Path(config["source_plan"]).read_text())
    model = {key: plan["model"][key] for key in ("id", "revision", "precision")}
    if model != {
        "id": config["model_id"],
        "revision": config["model_revision"],
        "precision": "bf16",
    }:
        raise ValueError("Source plan model differs")
    cases = config["cases"]
    if [case["label"] for case in cases] != ["checkpoint-1", "checkpoint-2"]:
        raise ValueError("Exactly two distinct checkpoint arms required")
    for case in cases:
        checkpoint = Path(case["checkpoint"])
        for filename in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json"):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(checkpoint / filename)
        if (
            hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
            != case["initial_adapter_sha256"]
        ):
            raise ValueError("Accepted checkpoint identity differs")
        if (
            json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
            != case["initial_step"]
        ):
            raise ValueError("Accepted checkpoint step differs")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    options = policy_template_options(plan)
    ids = prompt_token_ids(tokenizer, selected[0]["prompt"], template_options=options)
    request = {
        "row_id": config["row_id"],
        "model": model,
        "prompt_ids": ids,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "full_completion_ceiling": config["sampling"]["max_tokens"],
        "max_model_len": config["engine"]["max_model_len"],
        "template_options": options,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    }
    validate_request(request)
    write_new(out / "request.json", request)
    plan["judge_operation"] = {"id": config["run_id"], "cap_usd": config["judge_operation_cap_usd"]}
    write_new(out / "judge-plan.json", plan)
    for case in cases:
        resolved = dict(config, **case, request=str(out / "request.json"))
        write_new(out / f"{case['label']}-config.json", resolved)
    write_new(out / "prepared.json", config)


def summarize_metrics(
    left: dict[str, Any], right: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    if any(row["draws"] != 4 or row["unknown_draws"] for row in (left, right)):
        raise ValueError("Four known grades per arm required")
    if left["row_id"] != right["row_id"] or left["eligible_gt_ids"] != right["eligible_gt_ids"]:
        raise ValueError("Held-out scope or eligible GT denominator differs")
    if not left["eligible_gt_ids"]:
        raise ValueError("Nonempty eligible GT denominator required")
    reports = [
        {
            "model": "same-pinned-model",
            "sampling": "same-matched-settings",
            "rows": [{"split": "evaluation", "row_id": row["row_id"], "metrics": row}],
        }
        for row in (left, right)
    ]
    decision = compare(*reports, thresholds)
    return {
        "coverage_delta": right["known_gt_coverage"] - left["known_gt_coverage"],
        "matched_hits_delta": right["matched_hits"] - left["matched_hits"],
        "unmatched_findings_delta": right["unmatched_findings"] - left["unmatched_findings"],
        "quality_gain_established": False,
        "comparison": decision,
    }


def finish_evaluation(config: dict[str, Any], out: Path) -> dict[str, Any]:
    validate_evaluation_config(config)
    if json.loads((out / "prepared.json").read_text()) != config:
        raise ValueError("Prepared evaluation contract changed")
    request = json.loads((out / "request.json").read_text())
    validate_request(request)
    arms = []
    for case in config["cases"]:
        label = case["label"]
        expected = dict(config, **case, request=str(out / "request.json"))
        if json.loads((out / f"{label}-config.json").read_text()) != expected:
            raise ValueError("Checkpoint arm configuration changed")
        version = f"step-{case['initial_step']}:{case['initial_adapter_sha256']}"
        samples = json.loads((out / label / "samples.json").read_text())
        if len(samples) != 4 or any(
            sample["row_id"] != config["row_id"]
            or sample["prompt_ids"] != request["prompt_ids"]
            or sample["weight_version"] != version
            or sample["logprob_semantics"] != "sampling_processed"
            or len(sample["completion_ids"]) != len(sample["logprobs"])
            for sample in samples
        ):
            raise ValueError("Samples differ from matched evaluation contract")
        grading = out / f"{label}-grading"
        measured = json.loads((grading / "metrics.json").read_text())
        rewards = json.loads((grading / "rewards.json").read_text())
        if measured["row_id"] != config["row_id"]:
            raise ValueError("Graded scope differs from evaluation input")
        if len(rewards) != 4 or any(value is None for value in rewards):
            raise ValueError("Four known rewards required")
        arms.append(
            {
                "label": label,
                "weight_version": version,
                "metrics": measured,
                "rewards": rewards,
                "mean_reward": sum(rewards) / 4,
                "truncated_draws": sum(sample["finish_reason"] == "length" for sample in samples),
                "completion_tokens": sum(len(sample["completion_ids"]) for sample in samples),
            }
        )
    result = {
        "status": "matched_held_out_comparison_completed",
        "row_id": config["row_id"],
        "arms": arms,
        **summarize_metrics(arms[0]["metrics"], arms[1]["metrics"], config["thresholds"]),
        "mean_reward_delta": arms[1]["mean_reward"] - arms[0]["mean_reward"],
        "optimizer_steps": 0,
        "scope": "one held-out project, four draws per checkpoint; latest-update comparison",
    }
    write_new(out / "comparison.json", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
