"""Prepare and finish one free-text Solidity benchmark evaluation of one project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import benchmark_free_text as free_text
from benchmark_contract import grading_input_hashes, load_assets
from generation import validate_request
from grpo.samples import write_new
from runtime_io import read_json, seconds_until, sha256_bytes, sha256_file

# Settings a frozen base run must share with the trained run compared against it.
BASE_REFERENCE_KEYS = (
    "model_id",
    "model_revision",
    "base_precision",
    "sampling",
    "engine",
    "template_options",
    "benchmark_judge",
    "manifest_sha256",
    "row_id",
    "benchmark_suite",
    "output_format",
    "prompt_token_count",
    "sampler_backend",
    "rollout_image",
    "assets",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.phase != "settings" and args.out is None:
        parser.error(f"{args.phase} requires --out")
    for line in PHASES[args.phase](read_json(args.config), args.out):
        print(line)
    return 0


def phase_settings(config: dict[str, Any], _out: Path | None) -> list[str]:
    """Lines read by launch-evaluation.sh: images, time limits, arms and grading mode."""
    free_text.validate_config(config)
    return [
        config["native_image"],
        config["rollout_image"],
        str(config["seconds_limit"]),
        str(config["judge_seconds_limit"]),
        " ".join(config["generation_order"]),
        "overlap" if config["overlap_grading"] else "serial",
    ]


def phase_prepare(config: dict[str, Any], out: Path) -> list[str]:
    prepare_benchmark(config, out)
    return []


def phase_finish(config: dict[str, Any], out: Path) -> list[str]:
    finish_benchmark(config, out)
    return []


PHASES = {"settings": phase_settings, "prepare": phase_prepare, "finish": phase_finish}


# --- Preparation


def prepare_benchmark(config: dict[str, Any], out: Path) -> None:
    """Freeze the prompt tokens, private truth and judge prompt, and one config per arm."""
    free_text.validate_config(config)
    if (out / "prepared.json").exists():
        raise FileExistsError("Benchmark output must be fresh")
    if seconds_until(config["absolute_deadline"]) < config["seconds_limit"]:
        raise ValueError("Insufficient remaining machine budget")
    root = Path(config["assets"])
    if sha256_file(root / "BENCHMARK_MANIFEST.json") != config["manifest_sha256"]:
        raise ValueError("Benchmark manifest identity differs")
    lane, name = config["row_id"].split("/", 1)
    assets = load_assets(root, lane, name)
    if not assets["repo"].get("require_complete_gt_scope"):
        raise ValueError("Full native context requires reviewed complete GT source scope")
    verify_checkpoints(config["cases"])
    # The tokenizer comes from the GPU images; CPU tooling imports this module without it.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    options = config["template_options"]
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": assets["prompt"]}], **options
    )
    ids = list(tokens["input_ids"] if hasattr(tokens, "keys") else tokens)
    request = {
        "row_id": config["row_id"],
        "model": {
            "id": config["model_id"],
            "revision": config["model_revision"],
            "precision": "bf16",
        },
        "prompt_ids": ids,
        "prompt_ids_sha256": sha256_bytes(json.dumps(ids).encode()),
        "full_completion_ceiling": completion_allowance(config, len(ids)),
        "max_model_len": config["engine"]["max_model_len"],
        "template_options": options,
        "prompt_sha256": sha256_bytes(assets["prompt"].encode()),
        "chat_template_sha256": sha256_bytes(tokenizer.chat_template.encode()),
    }
    validate_request(request)
    (out / "prompt.txt").write_text(assets["prompt"])
    write_new(out / "request.json", request)
    # Policy requests never carry the separate private truth payload.
    write_new(out / "private-truth.json", assets["ground_truth"])
    (out / "judge-prompt.txt").write_text(assets["judge_prompt"])
    write_new(
        out / "source-scope.json",
        [{key: row[key] for key in ("path", "sha256")} for row in assets["sources"]],
    )
    if config.get("base_reference"):
        load_base_reference(config, out)
    for case in config["cases"]:
        write_new(
            out / f"{case['label']}-config.json",
            dict(config, **case, request=str(out / "request.json")),
        )
    write_new(out / "prepared.json", config)


def completion_allowance(config: dict[str, Any], prompt_token_count: int) -> int:
    """The output allowance is the whole context left after the measured prompt."""
    allowance = config["sampling"]["max_tokens"]
    if prompt_token_count != config["prompt_token_count"]:
        raise ValueError("Measured prompt token count changed")
    if allowance != config["engine"]["max_model_len"] - prompt_token_count:
        raise ValueError("Completion allowance must use the remaining context")
    return allowance


def verify_checkpoints(cases: list[dict[str, Any]]) -> None:
    """Every trained arm's adapter and step are the declared ones; the base arm has none."""
    for case in cases:
        if "checkpoint" not in case:
            continue
        checkpoint = Path(case["checkpoint"])
        if sha256_file(checkpoint / "adapter_model.safetensors") != case["initial_adapter_sha256"]:
            raise ValueError("Accepted checkpoint identity differs")
        if read_json(checkpoint / "trainer_state.json")["global_step"] != case["initial_step"]:
            raise ValueError("Checkpoint trainer state differs")


# --- Comparison


def finish_benchmark(config: dict[str, Any], out: Path) -> None:
    """Check each arm's samples and grades against the frozen inputs, then compare the arms."""
    free_text.validate_config(config)
    if read_json(out / "prepared.json") != config:
        raise ValueError("Frozen benchmark configuration changed")
    request = read_json(out / "request.json")
    validate_request(request)
    truth_data = (out / "private-truth.json").read_bytes()
    template_data = (out / "judge-prompt.txt").read_bytes()
    arms = []
    for case in config["cases"]:
        label = case["label"]
        expected = dict(config, **case, request=str(out / "request.json"))
        if read_json(out / f"{label}-config.json") != expected:
            raise ValueError("Frozen generation arm changed")
        sample_data = (out / label / "samples.json").read_bytes()
        samples = json.loads(sample_data)
        version = (
            f"base:{config['model_revision']}"
            if label == "base"
            else f"step-{case['initial_step']}:{case['initial_adapter_sha256']}"
        )
        if len(samples) != config["sampling"]["n"] or any(
            row["prompt_ids"] != request["prompt_ids"] or row["weight_version"] != version
            for row in samples
        ):
            raise ValueError("Matched sample input/weight identity differs")
        arm = read_json(out / f"{label}-grading/metrics.json")
        if (
            arm.get("input_sha256") != grading_input_hashes(sample_data, truth_data, template_data)
            or arm.get("judge") != config["benchmark_judge"]
        ):
            raise ValueError("Grading receipt differs from benchmark inputs or judge contract")
        arm.update(
            label=label,
            weight_version=version,
            completion_tokens=sum(len(row["completion_ids"]) for row in samples),
        )
        arms.append(arm)
    if config.get("base_reference"):
        arms.insert(0, {"label": "base", "draws": load_base_reference(config, out)})
    result = {
        **free_text.summarize(arms),
        "scope": f"{config['row_id']}; free text, {config['sampling']['n']} draws per arm",
        "output_format": "free_text",
        "draws": {arm["label"]: arm["draws"] for arm in arms},
    }
    if config.get("base_reference"):
        result["base_reference"] = config["base_reference"]
    write_new(out / "comparison.json", result)


def load_base_reference(config: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    """The frozen base run's graded draws, once it is shown to differ only in the weights."""
    reference = config["base_reference"]
    root = Path(reference["run_root"])
    saved = read_json(root / "prepared.json")
    for key in BASE_REFERENCE_KEYS:
        if saved.get(key) != config.get(key):
            raise ValueError(f"Frozen base reference differs: {key}")
    for name in ("private-truth.json", "judge-prompt.txt", "source-scope.json"):
        if (root / name).read_bytes() != (out / name).read_bytes():
            raise ValueError(f"Frozen base reference input differs: {name}")
    if (
        read_json(root / "request.json")["prompt_ids_sha256"]
        != read_json(out / "request.json")["prompt_ids_sha256"]
    ):
        raise ValueError("Frozen base reference policy prompt differs")
    data = (root / "comparison.json").read_bytes()
    if sha256_bytes(data) != reference["comparison_sha256"]:
        raise ValueError("Frozen base comparison identity differs")
    draws = json.loads(data)["draws"]["base"]
    if len(draws) != config["sampling"]["n"]:
        raise ValueError("Frozen base draw count differs from this evaluation")
    return draws


if __name__ == "__main__":
    raise SystemExit(main())
