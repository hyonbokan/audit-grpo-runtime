"""Bounded runtime-LoRA versus normally loaded merged-model measurements."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any


def phase_timings(metrics: dict[str, float], tokens: int) -> dict[str, float]:
    queued, scheduled, first, last = (
        metrics[k]
        for k in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")
    )
    if not (0 < queued <= scheduled <= first < last) or tokens < 2:
        raise ValueError("Missing or invalid engine timing/token count")
    return {
        "queue_seconds": scheduled - queued,
        "prefill_to_first_token_seconds": first - scheduled,
        "decode_seconds": last - first,
        "decode_tokens_per_second": (tokens - 1) / (last - first),
    }


def fixed_sequence_logprobs(
    rows: list[Any], prefix_length: int, tokens: list[int]
) -> list[float]:
    if rows is None or len(rows) != prefix_length + len(tokens):
        raise ValueError("Fixed-sequence prompt logprobs are not aligned")
    scores = []
    for row, token in zip(rows[prefix_length:], tokens, strict=True):
        if row is None or token not in row:
            raise ValueError("Fixed-sequence chosen token logprobs are not aligned")
        scores.append(float(row[token].logprob))
    return scores


def numerical_agreement(
    left: list[float], right: list[float], atol: float, rms_limit: float
) -> dict[str, Any]:
    if (
        not left
        or len(left) != len(right)
        or not all(math.isfinite(x) for x in left + right)
    ):
        raise ValueError("Fixed-sequence scores must be finite, nonempty and aligned")
    errors = [abs(a - b) for a, b in zip(left, right, strict=True)]
    rms = math.sqrt(sum(x * x for x in errors) / len(errors))
    maximum = max(errors)
    return {
        "passed": maximum <= atol and rms <= rms_limit,
        "max_absolute_logprob_difference": maximum,
        "rms_logprob_difference": rms,
        "atol": atol,
        "rms_limit": rms_limit,
        "absolute_differences": errors,
        "positions_exceeding_atol": [i for i, x in enumerate(errors) if x > atol],
    }


def run_arm(config: dict[str, Any], arm: str, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "status": "starting",
        "arm": arm,
        "measurements": [],
        "fixed_sequences": [],
        "judge_calls": 0,
    }
    try:
        from vllm import LLM, SamplingParams

        receipt["import_seconds"] = time.monotonic() - started
        options = dict(config["engine"])
        options["disable_log_stats"] = False
        request_options = {}
        if arm == "adapter":
            import msgspec
            from adapter_mapping import prepare_adapter
            from vllm.lora.request import LoRARequest

            receipt["mapping"] = prepare_adapter(
                Path(config["adapter_path"]), out / "mapped-adapter"
            )
            options.update(config["lora_engine"])
            options["worker_extension_cls"] = "adapter_compat.VerifiedAdapterWorker"
            model_path = config["base_path"]
        elif arm == "merged":
            options["enable_lora"] = False
            model_path = config["merged_path"]
            manifest = json.loads(
                (Path(model_path) / "MERGE_MANIFEST.json").read_text()
            )
            if manifest["status"] != "merged_and_verified":
                raise ValueError("Merged artifact is not complete/verified")
            if manifest["adapter_sha256"] != config["adapter_sha256"]:
                raise ValueError("Merged adapter identity differs")
        else:
            raise ValueError("Unknown arm")
        receipt["resolved_engine"] = options
        load_start = time.monotonic()
        engine = LLM(model=model_path, tokenizer=config["base_path"], **options)
        receipt["engine_startup_seconds"] = time.monotonic() - load_start
        if arm == "adapter":
            adapter_start = time.monotonic()
            request = LoRARequest(
                "checkpoint2-diagnostic",
                2,
                str(out / "mapped-adapter"),
                load_inplace=True,
            )
            engine.collective_rpc("load_adapter", args=(msgspec.to_builtins(request),))
            reports = engine.collective_rpc(
                "verify_adapter",
                args=(2, config["adapter_path"], config["adapter_sha256"]),
            )
            (out / "adapter-transfer.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )
            if len(reports) != 1 or not reports[0]["cast_transfer_verified"]:
                raise ValueError("Runtime adapter transfer verification failed")
            receipt["adapter_setup_seconds"] = time.monotonic() - adapter_start
            request_options["lora_request"] = request
        receipt["ready_seconds"] = time.monotonic() - started
        sample = SamplingParams(**config["sampling"])
        warm_start = time.monotonic()
        engine.generate(
            [{"prompt_token_ids": config["prompts"][0]["token_ids"]}],
            sample,
            use_tqdm=False,
            **request_options,
        )
        receipt["discarded_warmup_seconds"] = time.monotonic() - warm_start
        for prompt in config["prompts"]:
            for repeat in range(config["repetitions"]):
                begin = time.monotonic()
                output = engine.generate(
                    [{"prompt_token_ids": prompt["token_ids"]}],
                    sample,
                    use_tqdm=False,
                    **request_options,
                )[0]
                elapsed = time.monotonic() - begin
                tokens = list(output.outputs[0].token_ids)
                if len(tokens) != config["sampling"]["max_tokens"]:
                    raise ValueError(
                        "Bounded generation output allowance was not fully exercised"
                    )
                if (
                    output.prompt_token_ids != prompt["token_ids"]
                    or output.metrics is None
                ):
                    raise ValueError(
                        "Prompt identity or engine timing metrics unavailable"
                    )
                metrics = dataclasses.asdict(output.metrics)
                measurement = {
                    "prompt": prompt["name"],
                    "repeat": repeat,
                    "prompt_tokens": len(prompt["token_ids"]),
                    "prompt_sha256": prompt["sha256"],
                    "completion_ids": tokens,
                    "generation_seconds": elapsed,
                    "engine_metrics": metrics,
                    "finish_reason": output.outputs[0].finish_reason,
                    **phase_timings(metrics, len(tokens)),
                }
                receipt["measurements"].append(measurement)
                (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "event": "timed_request",
                            "arm": arm,
                            "prompt": prompt["name"],
                            "repeat": repeat,
                            "decode_tokens_per_second": measurement[
                                "decode_tokens_per_second"
                            ],
                        }
                    ),
                    flush=True,
                )
            fixed = config["fixed_completion_ids"]
            teacher_settings = dict(config["sampling"])
            teacher_settings.update(max_tokens=1, prompt_logprobs=0)
            scored = engine.generate(
                [{"prompt_token_ids": prompt["token_ids"] + fixed}],
                SamplingParams(**teacher_settings),
                use_tqdm=False,
                **request_options,
            )[0]
            chosen = fixed_sequence_logprobs(
                scored.prompt_logprobs, len(prompt["token_ids"]), fixed
            )
            receipt["fixed_sequences"].append(
                {
                    "prompt": prompt["name"],
                    "token_ids": fixed,
                    "chosen_raw_logprobs": chosen,
                    "scoring_method": "prompt_logprobs=0, chosen causal suffix tokens",
                    "semantics": "raw normalized model logprobs before sampling filters",
                }
            )
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["total_seconds"] = time.monotonic() - started
        (out / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )


def compare(config: dict[str, Any], root: Path) -> dict[str, Any]:
    arms = {
        arm: json.loads((root / arm / "receipt.json").read_text())
        for arm in ("adapter", "merged")
    }
    if any(value["status"] != "completed" for value in arms.values()):
        raise ValueError("Both arms must complete before comparison")
    expected_prompts = {p["name"]: p["sha256"] for p in config["prompts"]}
    for value in arms.values():
        if (
            len(value["fixed_sequences"]) != len(expected_prompts)
            or len(value["measurements"])
            != len(expected_prompts) * config["repetitions"]
        ):
            raise ValueError("Incomplete fixed-sequence or timed-request evidence")
        if {s["prompt"] for s in value["fixed_sequences"]} != set(expected_prompts):
            raise ValueError(
                "Fixed-sequence prompt identities differ from configuration"
            )
        expected = {
            (name, i) for name in expected_prompts for i in range(config["repetitions"])
        }
        actual = {(m["prompt"], m["repeat"]) for m in value["measurements"]}
        if expected != actual or any(
            m["prompt_sha256"] != expected_prompts[m["prompt"]]
            for m in value["measurements"]
        ):
            raise ValueError("Timed-request identities differ from configuration")
    agreement = []
    for left, right in zip(
        arms["adapter"]["fixed_sequences"],
        arms["merged"]["fixed_sequences"],
        strict=True,
    ):
        if left["prompt"] != right["prompt"] or left["token_ids"] != right["token_ids"]:
            raise ValueError("Fixed token sequences differ")
        agreement.append(
            {
                "prompt": left["prompt"],
                **numerical_agreement(
                    left["chosen_raw_logprobs"],
                    right["chosen_raw_logprobs"],
                    config["logprob_atol"],
                    config["logprob_rms_limit"],
                ),
            }
        )
    timing = []
    for left, right in zip(
        arms["adapter"]["measurements"], arms["merged"]["measurements"], strict=True
    ):
        if (left["prompt_sha256"], left["repeat"]) != (
            right["prompt_sha256"],
            right["repeat"],
        ):
            raise ValueError("Timing prompt/repetition differs")
        timing.append(
            {
                "prompt": left["prompt"],
                "repeat": left["repeat"],
                "adapter_decode_tokens_per_second": left["decode_tokens_per_second"],
                "merged_decode_tokens_per_second": right["decode_tokens_per_second"],
                "decode_speedup": right["decode_tokens_per_second"]
                / left["decode_tokens_per_second"],
                "adapter_prefill_seconds": left["prefill_to_first_token_seconds"],
                "merged_prefill_seconds": right["prefill_to_first_token_seconds"],
            }
        )
    stability = []
    for arm, value in arms.items():
        for prompt in expected_prompts:
            rates = [
                m["decode_tokens_per_second"]
                for m in value["measurements"]
                if m["prompt"] == prompt
            ]
            spread = (max(rates) - min(rates)) / (sum(rates) / len(rates))
            stability.append(
                {
                    "arm": arm,
                    "prompt": prompt,
                    "relative_spread": spread,
                    "passed": spread <= config["max_relative_decode_rate_spread"],
                }
            )
    numerical_passed = all(item["passed"] for item in agreement)
    stability_passed = all(item["passed"] for item in stability)
    preservation_path = root / "preservation-after.json"
    preservation_passed = (
        preservation_path.is_file()
        and json.loads(preservation_path.read_text())["status"] == "preserved"
    )
    result = {
        "status": "comparison_completed",
        "numerical_agreement_passed": numerical_passed,
        "repeat_stability_passed": stability_passed,
        "acceptance_passed": numerical_passed
        and stability_passed
        and preservation_passed,
        "source_preservation_passed": preservation_passed,
        "repeat_stability": stability,
        "agreement": agreement,
        "timing": timing,
        "judge_calls": 0,
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("adapter", "merged", "compare"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.phase == "compare":
        compare(config, args.out)
    else:
        run_arm(config, args.phase, args.out)


if __name__ == "__main__":
    main()
