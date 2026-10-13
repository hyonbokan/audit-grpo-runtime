"""Standalone BF16 generation benchmark; never grades or updates weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--context", type=int, default=36864)
    run = commands.add_parser("run")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--eager", action="store_true")
    run.add_argument("--telemetry", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        request = json.loads(args.prepared.read_text())
        validate_request(request)
        run_benchmark(args, request)
        return
    plan = json.loads(args.plan.read_text())
    sample = json.loads(args.sample.read_text())
    ids = sample["prompt_ids"]
    if plan["model"]["precision"] != "bf16":
        raise ValueError("Benchmark must preserve BF16 weights")
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Nonempty native prompt tokens required")
    if len(ids) + plan["training"]["max_completion_tokens"] > args.context:
        raise ValueError("Context must accommodate the whole prompt and completion ceiling")
    request = {
        "model": plan["model"],
        "row_id": sample["row_id"],
        "profile": plan["sampling_profile"],
        "prompt_ids": ids,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "source_sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "source_plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "max_model_len": args.context,
        "full_completion_ceiling": plan["training"]["max_completion_tokens"],
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 512,
            "logprobs": 1,
            "seed": 3407,
        },
        "full_completion_budget_tested": False,
        "judge_calls": 0,
        "optimizer_steps": 0,
    }
    validate_request(request)
    with args.out.open("x") as stream:
        json.dump(request, stream, indent=2)
        stream.write("\n")


def validate_request(request):
    ids = request["prompt_ids"]
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != request["prompt_ids_sha256"]:
        raise ValueError("Native prompt hash differs")
    if request["model"]["precision"] != "bf16":
        raise ValueError("BF16 weights required")
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Nonempty native prompt tokens required")
    if len(ids) + request["full_completion_ceiling"] > request["max_model_len"]:
        raise ValueError("Context must accommodate the whole prompt and completion ceiling")


def capture_batch(
    output,
    prompt_ids,
    *,
    count,
    seconds,
    logprob_semantics: str | None = None,
    weight_version: str | None = None,
    require_sampling_support: bool = False,
):
    if require_sampling_support and logprob_semantics != "sampling_processed":
        raise ValueError("Sampling support replay requires processed probabilities")
    if logprob_semantics is not None:
        if logprob_semantics not in ("model_raw", "sampling_processed"):
            raise ValueError("Unknown probability semantics")
        if not isinstance(weight_version, str) or not weight_version:
            raise ValueError("Explicit policy weight version required")
    if list(output.prompt_token_ids) != prompt_ids:
        raise ValueError("Engine prompt differs from complete native prompt")
    if len(output.outputs) != count:
        raise ValueError("Engine completion count differs")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Positive measured duration required")
    draws = []
    for draw in output.outputs:
        ids = list(draw.token_ids)
        probabilities = draw.logprobs
        if probabilities is None or len(probabilities) != len(ids):
            raise ValueError("Missing token logprob records")
        try:
            values = [entry[token].logprob for token, entry in zip(ids, probabilities, strict=True)]
        except KeyError as exc:
            raise ValueError("Missing chosen-token logprob") from exc
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Nonfinite chosen-token logprob")
        draws.append(
            {
                "index": draw.index,
                "raw_text": draw.text,
                "completion_ids": ids,
                "logprobs": values,
                "finish_reason": draw.finish_reason,
                "stop_reason": draw.stop_reason,
            }
        )
        if logprob_semantics is not None:
            mask = getattr(draw, "sampling_mask", None)
            support = None if mask is None else [list(tokens) for tokens in mask.token_ids]
            if support is None:
                if require_sampling_support:
                    raise ValueError("Required sampling support was not returned by the engine")
            else:
                if len(support) != len(ids):
                    raise ValueError("Per-token sampling support lengths differ from completion")
                if any(
                    not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or len(set(tokens)) != len(tokens)
                    or chosen not in tokens
                    for chosen, tokens in zip(ids, support, strict=True)
                ):
                    raise ValueError("Invalid per-token sampling support")
            draws[-1].update(
                logprob_semantics=logprob_semantics,
                weight_version=weight_version,
                raw_logprobs=values if logprob_semantics == "model_raw" else None,
                sampling_support_ids=support,
            )
    tokens = sum(len(draw["completion_ids"]) for draw in draws)
    return {
        "draw_count": count,
        "prompt_tokens": len(prompt_ids),
        "draws": draws,
        "generation_seconds_including_prefill": seconds,
        "completion_tokens": tokens,
        "aggregate_tokens_per_second": tokens / seconds,
        "logprob_distribution_verified": False,
    }


def run_benchmark(args, request):
    import importlib.metadata
    import time

    args.out.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "loading",
        "request_sha256": hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        "model": request["model"],
        "row_id": request["row_id"],
        "prompt_tokens": len(request["prompt_ids"]),
        "prompt_ids_sha256": request["prompt_ids_sha256"],
        "sampling": request["sampling"],
        "model_copy": "untouched base; no active LoRA adapter",
        "judge_calls": 0,
        "optimizer_steps": 0,
        "batches": [],
        "full_completion_budget_tested": False,
        "backward_tested": False,
        "logprob_distribution_verified": False,
        "trainer_integration_verified": False,
        "engine": {
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "max_model_len": request["max_model_len"],
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 4096,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "language_model_only": True,
            "enforce_eager": args.eager,
            "seed": 3407,
        },
    }
    result_path = args.out / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    try:
        from vllm import LLM, SamplingParams

        result["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers", "triton", "unsloth", "trl", "peft")
        }
        config = json.loads((args.model_dir / "config.json").read_text())
        if config.get("_name_or_path") not in (None, request["model"]["id"]):
            raise ValueError("Local model identity differs")
        if args.model_dir.name != request["model"]["revision"]:
            raise ValueError("Local snapshot directory must be the pinned revision")
        engine = LLM(model=str(args.model_dir), tokenizer=str(args.model_dir), **result["engine"])
        result["model_load_seconds"] = time.monotonic() - started
        prompt = {"prompt_token_ids": request["prompt_ids"]}
        warmup = dict(request["sampling"], max_tokens=32)
        engine.generate([prompt], SamplingParams(n=1, **warmup), use_tqdm=False)
        result["status"] = "benchmarking"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        for count in (1, 2, 4):
            before = time.monotonic()
            outputs = engine.generate(
                [prompt], SamplingParams(n=count, **request["sampling"]), use_tqdm=False
            )
            duration = time.monotonic() - before
            if len(outputs) != 1:
                raise ValueError("Expected one complete input prompt")
            batch = capture_batch(outputs[0], request["prompt_ids"], count=count, seconds=duration)
            batch["estimated_compute_usd"] = duration / 3600 * 2.51
            (args.out / f"batch-{count}.json").write_text(json.dumps(batch, indent=2) + "\n")
            result["batches"].append({key: value for key, value in batch.items() if key != "draws"})
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        result["status"] = "short_generation_benchmark_completed"
    except Exception as exc:
        result["status"] = "benchmark_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["total_seconds"] = time.monotonic() - started
        result["estimated_compute_usd"] = result["total_seconds"] / 3600 * 2.51
        result["gpu_telemetry_path"] = str(args.telemetry)
        result_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
