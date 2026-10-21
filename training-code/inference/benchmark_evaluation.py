"""Prepare frozen native Solidity comparisons and summarize benchmark grading."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import benchmark_free_text as free_text
from benchmark_contract import grading_input_hashes, load_assets, parse_completion
from generation import validate_request
from grpo.samples import write_new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "normalize", "finish"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate_benchmark_config(config)
    if args.phase == "prepare":
        prepare_benchmark(config, args.out)
    elif args.phase == "normalize":
        label = args.label or config["cases"][0]["label"]
        if label not in {case["label"] for case in config["cases"]}:
            raise ValueError("Normalization arm is outside the declared comparison")
        samples, report = normalize_samples(
            json.loads((args.out / label / "samples.json").read_text())
        )
        destination = args.out / "conditional" / label
        destination.mkdir(parents=True, exist_ok=False)
        write_new(destination / "samples.json", samples)
        write_new(destination / "strict-validity.json", report)
        if config.get("continuation_reference"):
            write_new(args.out / "strict-validity.json", report)
    else:
        finish_benchmark(config, args.out)
    return 0


def validate_benchmark_config(config: dict[str, Any]) -> None:
    if config.get("benchmark_suite") == free_text.SUITE:
        free_text.validate_config(config)
        return
    scope = json.loads(Path(__file__).parents[1].joinpath("benchmark_scope.json").read_text())
    limits = scope.pop("continuation")
    suite = scope.pop("solidity1")
    scope.pop("free_text")
    markdown = config.get("benchmark_suite") == "solidity1_markdown"
    profile = config.get("evaluation_profile")
    if profile not in (None, "full_native_context"):
        raise ValueError("Unknown evaluation profile")
    native = None
    weekend = bool(config.get("weekend_schedule_id"))
    if weekend:
        # Imported only for controller evaluations; an unregistered schedule id fails here.
        import schedules

        reviewed = schedules.scope(config)
    if profile == "full_native_context":
        if not markdown or not config.get("merged_model_path"):
            raise ValueError("Full native context requires matched Markdown merged evaluation")
        native = json.loads(
            Path(__file__).parents[1].joinpath("native_context_scope.json").read_text()
        )
        if weekend:
            native.update(
                absolute_deadline=reviewed["final_deadline"],
                seconds_limit=reviewed["evaluation_seconds_limit"],
                judge_operation_cap_usd=reviewed["evaluation_judge_cap_usd"],
            )
            scope["template_options"]["reasoning_effort"] = reviewed["evaluation_effort"]
    elif weekend:
        raise ValueError("Weekend evaluation requires full native context")
    if config.get("benchmark_suite") not in (None, "solidity1", "solidity1_markdown"):
        raise ValueError("Unknown benchmark suite")
    if config.get("benchmark_suite") in ("solidity1", "solidity1_markdown"):
        if config.get("row_id") not in suite["row_ids"]:
            raise ValueError("Project is outside the reviewed Solidity1 suite")
        context = config.get("engine", {}).get("max_model_len")
        maximum_context = native["max_model_len"] if native else suite["maximum_context"]
        if type(context) is not int or not 32768 <= context <= maximum_context:
            raise ValueError("Solidity1 context exceeds the reviewed bound")
        scope.update(
            row_id=config["row_id"],
            engine=dict(scope["engine"], max_model_len=context),
            **{
                key: suite[key]
                for key in (
                    "seconds_limit",
                    "judge_seconds_limit",
                    "judge_operation_cap_usd",
                    "content_panel",
                )
            },
        )
        if markdown:
            scope.pop("content_panel")
            scope.update(output_format="markdown", sampling=dict(scope["sampling"], n=1))
            if config.get("content_panel") or config.get("continuation_reference"):
                raise ValueError("Markdown one-shot requires fresh matched arms")
        if config.get("grading_concurrency", 1) not in (1, 4):
            raise ValueError("Solidity1 grading concurrency must be one or four")
        if type(config.get("overlap_grading", False)) is not bool:
            raise ValueError("Solidity1 grading overlap must be an explicit boolean")
    if native:
        prompt_count = config.get("prompt_token_count")
        if type(prompt_count) is not int or not 0 < prompt_count < native["max_model_len"]:
            raise ValueError("Valid measured prompt token count required")
        engine = dict(scope["engine"], **native["engine_overrides"])
        engine["max_model_len"] = native["max_model_len"]
        for key in native["removed_engine_options"]:
            engine.pop(key, None)
        scope.update(
            engine=engine,
            sampling=dict(scope["sampling"], max_tokens=native["max_model_len"] - prompt_count),
            **{
                key: native[key]
                for key in (
                    "absolute_deadline",
                    "seconds_limit",
                    "judge_seconds_limit",
                    "judge_operation_cap_usd",
                )
            },
        )
    continuation = bool(config.get("continuation_reference"))
    repeat = config.get("repeat_evaluation")
    if repeat is not None:
        if not native or not isinstance(repeat, dict) or set(repeat) != {"seed", "arms"}:
            raise ValueError("Repeat requires an explicit full-context seed and arms")
        seed, arms = repeat["seed"], repeat["arms"]
        if (
            type(seed) is not int
            or not 0 <= seed < 2**32
            or seed == scope["sampling"]["seed"]
            or not isinstance(arms, list)
            or not arms
            or any(arm not in {row.get("label") for row in config.get("cases", [])} for arm in arms)
            or len(set(arms)) != len(arms)
        ):
            raise ValueError("Repeat requires distinct declared arms and a fresh valid seed")
        scope["sampling"] = dict(scope["sampling"], seed=seed)
        scope["engine"] = dict(scope["engine"], seed=seed)
    if continuation:
        scope.update(
            judge_seconds_limit=limits["judge_seconds_limit"],
            judge_operation_cap_usd=limits["judge_operation_cap_usd"],
        )
    bounds = {"seconds_limit", "judge_seconds_limit", "judge_operation_cap_usd"}
    for key, expected in scope.items():
        if key in bounds:
            if type(config.get(key)) not in (int, float) or not 0 < config[key] <= expected:
                raise ValueError(f"Invalid benchmark budget: {key}")
        elif config.get(key) != expected:
            raise ValueError(f"Benchmark differs from reviewed contract: {key}")
    cases = config.get("cases", [])
    if "evaluation_deadline_override" in config:
        extension = (
            datetime.fromisoformat(config["evaluation_deadline_override"])
            - datetime.fromisoformat(config["absolute_deadline"])
        ).total_seconds()
        if not markdown or not config.get("merged_model_path") or not 0 < extension <= 3600:
            raise ValueError("Invalid merged evaluation deadline override")
    order = config.get("generation_order", [case["label"] for case in cases])
    if repeat is not None:
        valid_order = order == repeat["arms"]
    else:
        valid_order = len(order) == len(cases) and set(order) == {case["label"] for case in cases}
    if not valid_order:
        raise ValueError("Invalid generation order")
    if continuation:
        case = cases[0] if len(cases) == 1 else {}
        step = case.get("initial_step")
        if (
            type(step) is not int
            or not 3 <= step <= limits["maximum_step"]
            or case.get("label") != f"checkpoint-{step}"
            or not case.get("checkpoint")
            or not case.get("initial_adapter_sha256")
            or config.get("content_panel") != limits["content_panel"]
            or not config.get("reference_comparison_sha256")
        ):
            raise ValueError("One new checkpoint and a frozen conditional reference required")
    elif weekend:
        case = cases[-1] if len(cases) == 2 else {}
        step = case.get("initial_step")
        if (
            cases[0] != {"label": "base", "initial_step": 0}
            or type(step) is not int
            or not 2 <= step <= 512
            or case.get("label") != f"checkpoint-{step}"
            or not case.get("checkpoint")
            or not case.get("initial_adapter_sha256")
        ):
            raise ValueError("Weekend evaluation requires the actual final checkpoint identity")
    elif [(row.get("label"), row.get("initial_step")) for row in cases] != [
        ("base", 0),
        ("checkpoint-2", 2),
    ]:
        raise ValueError("Unchanged BF16 base and checkpoint2 arms required")
    if not continuation and any(
        key in cases[0] for key in ("checkpoint", "initial_adapter_sha256")
    ):
        raise ValueError("Base measurement must have no adapter")
    if "initial_step" in config and not any(
        all(
            config.get(key) == row.get(key)
            for key in ("label", "initial_step", "checkpoint", "initial_adapter_sha256")
        )
        for row in cases
    ):
        raise ValueError("Generation arm differs from declared comparison")


def completion_allowance(config: dict[str, Any], prompt_token_count: int) -> int:
    allowance = config["sampling"]["max_tokens"]
    if config.get("evaluation_profile") == "full_native_context" or (
        config.get("benchmark_suite") == free_text.SUITE
    ):
        if prompt_token_count != config.get("prompt_token_count"):
            raise ValueError("Measured prompt token count changed")
        if allowance != config["engine"]["max_model_len"] - prompt_token_count:
            raise ValueError("Completion allowance must use the remaining context")
    return allowance


def prepare_benchmark(config: dict[str, Any], out: Path) -> None:
    validate_benchmark_config(config)
    if (out / "prepared.json").exists():
        raise FileExistsError("Benchmark output must be fresh")
    deadline = datetime.fromisoformat(
        config.get("evaluation_deadline_override", config["absolute_deadline"])
    )
    if (deadline - datetime.now(UTC)).total_seconds() < config["seconds_limit"]:
        raise ValueError("Insufficient remaining machine budget")
    root = Path(config["assets"])
    if (
        hashlib.sha256((root / "BENCHMARK_MANIFEST.json").read_bytes()).hexdigest()
        != config["manifest_sha256"]
    ):
        raise ValueError("Benchmark manifest identity differs")
    lane, name = config["row_id"].split("/", 1)
    assets = load_assets(root, lane, name, output_format=config.get("output_format", "json"))
    full_context = config.get("evaluation_profile") == "full_native_context" or (
        config.get("benchmark_suite") == free_text.SUITE
    )
    if full_context and not assets["repo"].get("require_complete_gt_scope"):
        raise ValueError("Full native context requires reviewed complete GT source scope")
    verify_checkpoints(config["cases"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    options = config["template_options"]
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": assets["prompt"]}], **options
    )
    ids = list(tokens["input_ids"] if hasattr(tokens, "keys") else tokens)
    allowance = completion_allowance(config, len(ids))
    request = {
        "row_id": config["row_id"],
        "model": {
            "id": config["model_id"],
            "revision": config["model_revision"],
            "precision": "bf16",
        },
        "prompt_ids": ids,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "full_completion_ceiling": allowance,
        "max_model_len": config["engine"]["max_model_len"],
        "template_options": options,
        "prompt_sha256": hashlib.sha256(assets["prompt"].encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
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
    if config.get("continuation_reference"):
        load_reference(config, out)
    for case in config["cases"]:
        write_new(
            out / f"{case['label']}-config.json",
            dict(config, **case, request=str(out / "request.json")),
        )
    write_new(out / "prepared.json", config)


def verify_checkpoints(cases: list[dict[str, Any]]) -> None:
    """Every trained arm's adapter and step are the declared ones; the base arm has none."""
    for case in cases:
        if "checkpoint" not in case:
            continue
        checkpoint = Path(case["checkpoint"])
        if (
            hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
            != case["initial_adapter_sha256"]
        ):
            raise ValueError("Accepted checkpoint identity differs")
        if (
            json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
            != (case["initial_step"])
        ):
            raise ValueError("Checkpoint trainer state differs")


def summarize_benchmark(
    left: dict[str, Any], right: dict[str, Any], *, sample_count: int = 4
) -> dict[str, Any]:
    arms = [left, right]
    if sample_count not in (1, 4) or any(len(arm["draws"]) != sample_count for arm in arms):
        raise ValueError("Declared recorded draws per arm required")
    known = all(row["status"] == "graded" for arm in arms for row in arm["draws"])
    if known and len({row["known_gt_count"] for arm in arms for row in arm["draws"]}) != 1:
        raise ValueError("Matched private GT denominator required")
    return {
        "status": "matched_benchmark_completed" if known else "inconclusive",
        "mean_exact_coverage_delta": (
            sum(row["exact_coverage"] for row in right["draws"])
            - sum(row["exact_coverage"] for row in left["draws"])
        )
        / sample_count
        if known
        else None,
        "mean_coverage_including_partial_delta": (
            sum(row["coverage_including_partial"] for row in right["draws"])
            - sum(row["coverage_including_partial"] for row in left["draws"])
        )
        / sample_count
        if known
        else None,
        "quality_gain_established": False,
        "optimizer_steps": 0,
        "scope": "established Munchables regression benchmark; four draws per arm",
    }


def finish_benchmark(config: dict[str, Any], out: Path) -> None:
    validate_benchmark_config(config)
    if json.loads((out / "prepared.json").read_text()) != config:
        raise ValueError("Frozen benchmark configuration changed")
    request = json.loads((out / "request.json").read_text())
    validate_request(request)
    truth_data = (out / "private-truth.json").read_bytes()
    template_data = (out / "judge-prompt.txt").read_bytes()
    arms = []
    strict_validity = {}
    conditional = bool(config.get("continuation_reference") or config.get("content_panel"))
    for case in config["cases"]:
        label = case["label"]
        if config.get("repeat_evaluation") and label not in config["repeat_evaluation"]["arms"]:
            continue
        expected = dict(config, **case, request=str(out / "request.json"))
        if json.loads((out / f"{label}-config.json").read_text()) != expected:
            raise ValueError("Frozen generation arm changed")
        sample_root = out / "conditional" if conditional else out
        sample_data = (sample_root / label / "samples.json").read_bytes()
        samples = json.loads(sample_data)
        if config.get("benchmark_suite") == "solidity1":
            raw = json.loads((out / label / "samples.json").read_text())
            expected_samples, report = normalize_samples(raw)
            if samples != expected_samples:
                raise ValueError("Grading samples differ from the sole-fence normalization")
            strict_validity[label] = report["strict_valid_fraction"]
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
        arm = json.loads((out / f"{label}-grading/metrics.json").read_text())
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
    if config.get("benchmark_suite") == free_text.SUITE:
        result = {
            **free_text.summarize(arms),
            "scope": f"{config['row_id']}; free text, {config['sampling']['n']} draws per arm",
            "output_format": "free_text",
            "draws": {arm["label"]: arm["draws"] for arm in arms},
        }
    elif config.get("repeat_evaluation") and len(arms) == 1:
        result = {
            "status": "single_arm_completed"
            if all(draw["status"] == "graded" for draw in arms[0]["draws"])
            else "inconclusive",
            "arms": arms,
            "scope": f"{config['row_id']}; single-arm repeat, no new paired comparison",
            "sample_count": config["sampling"]["n"],
            "output_format": config["output_format"],
            "mean_exact_coverage_delta": None,
            "mean_coverage_including_partial_delta": None,
            "quality_gain_established": False,
            "optimizer_steps": 0,
        }
    elif config.get("continuation_reference"):
        reference = load_reference(config, out)
        previous = reference["arms"]["checkpoint-2"]
        base = reference["arms"]["base"]
        measured = arms[0]
        summary = summarize_benchmark(previous, measured)
        base_summary = summarize_benchmark(base, measured)
        known = summary["status"] == "matched_benchmark_completed"
        positive = (
            known
            and summary["mean_exact_coverage_delta"] > 0
            and summary["mean_coverage_including_partial_delta"] >= 0
        )
        result = {
            **summary,
            "panel": config["content_panel"],
            "positive_quality_signal": positive if known else None,
            "versus_base": base_summary,
            "strict_json_validity": {
                **reference["strict_json_validity"],
                config["cases"][0]["label"]: json.loads((out / "strict-validity.json").read_text())[
                    "strict_valid_fraction"
                ],
            },
            "arms": {"base": base, "checkpoint-2": previous, config["cases"][0]["label"]: measured},
            "reused_reference": config["continuation_reference"],
            "reference_comparison_sha256": config["reference_comparison_sha256"],
        }
    else:
        result = {**summarize_benchmark(*arms, sample_count=config["sampling"]["n"]), "arms": arms}
        if config.get("benchmark_suite") == "solidity1":
            result.update(
                scope=f"established {config['row_id']} regression benchmark; four draws per arm",
                panel=config["content_panel"],
                strict_json_validity=strict_validity,
            )
        elif config.get("benchmark_suite") == "solidity1_markdown":
            result.update(
                scope=f"{config['row_id']}; one Markdown audit per arm",
                output_format="markdown",
                sample_count=1,
            )
    if config.get("repeat_evaluation"):
        result["repeat_evaluation"] = config["repeat_evaluation"]
    write_new(out / "comparison.json", result)


def normalize_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = copy.deepcopy(samples)
    strict, removed = [], []
    for original, sample in zip(samples, normalized, strict=True):
        try:
            parse_completion(original["final_text"], original["finish_reason"])
            strict.append(True)
        except ValueError:
            strict.append(False)
        text = sample["final_text"].strip()
        changed = False
        if (
            sample["finish_reason"] == "stop"
            and text.startswith("```json\n")
            and text.endswith("\n```")
        ):
            inner = text[len("```json\n") : -len("\n```")]
            try:
                parse_completion(inner, "stop")
                sample["final_text"] = inner
                changed = True
            except ValueError:
                pass
        removed.append(changed)
    return normalized, {
        "strict_valid_fraction": sum(strict) / len(samples),
        "strict_valid_draws": strict,
        "fences_removed": removed,
    }


def load_reference(config: dict[str, Any], out: Path) -> dict[str, Any]:
    root = Path(config["continuation_reference"])
    saved = json.loads((root / "prepared.json").read_text())
    for key in (
        "model_id",
        "model_revision",
        "base_precision",
        "sampling",
        "engine",
        "template_options",
        "benchmark_judge",
        "manifest_sha256",
        "row_id",
    ):
        if saved[key] != config[key]:
            raise ValueError(f"Saved benchmark reference differs: {key}")
    for name in ("private-truth.json", "judge-prompt.txt", "source-scope.json"):
        if (root / name).read_bytes() != (out / name).read_bytes():
            raise ValueError(f"Saved benchmark reference input differs: {name}")
    saved_request = json.loads((root / "request.json").read_text())
    request = json.loads((out / "request.json").read_text())
    if saved_request["prompt_ids_sha256"] != request["prompt_ids_sha256"]:
        raise ValueError("Saved benchmark reference policy prompt differs")
    data = (root / "normalized/comparison.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != config["reference_comparison_sha256"]:
        raise ValueError("Saved comparison identity differs")
    reference = json.loads(data)
    if reference["panel"] != config["content_panel"]:
        raise ValueError("Saved benchmark content panel differs")
    return reference


if __name__ == "__main__":
    raise SystemExit(main())
