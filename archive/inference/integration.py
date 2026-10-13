"""Probe a real adapter and four full-budget BF16 vLLM outputs; no updates or judging."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from contextlib import ExitStack

from adapter_mapping import prepare_adapter
from benchmark import capture_batch, validate_request
from sampler_compat import native_engine_sampling


def check_checkpoint(step):
    metrics = json.loads((step / "step_metrics.json").read_text())
    if metrics.get("global_step") != 1 or metrics.get("adapter_weights_changed") is not True:
        raise ValueError("A verified changed-weight update is required")
    checkpoint = step / "checkpoints/checkpoint-1"
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(name)
    return checkpoint


def outputs_changed(first, second):
    a, b = first["draws"][0], second["draws"][0]
    return a["completion_ids"] != b["completion_ids"] or any(
        abs(x - y) > 1e-7 for x, y in zip(a["logprobs"], b["logprobs"], strict=True)
    )


def repeated_token_ids(first, second):
    return first["draws"][0]["completion_ids"] == second["draws"][0]["completion_ids"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return run_integration(args)


def run_integration(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "status": "preparing",
        "optimizer_updates": 0,
        "judge_calls": 0,
        "trainer_weight_sync_verified": False,
        "batches": [],
    }
    scope = ExitStack()
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from grpo.pilot_runner import validate_step

        checkpoint = check_checkpoint(args.step)
        if validate_step(args.step) != checkpoint:
            raise ValueError("Canonical checkpoint gate differs")
        request = json.loads(args.request.read_text())
        validate_request(request)
        if args.model.name != request["model"]["revision"]:
            raise ValueError("Pinned snapshot required")
        adapter_hash = hashlib.sha256(
            (checkpoint / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        config = json.loads((checkpoint / "adapter_config.json").read_text())
        if config["r"] != 16 or config["lora_alpha"] != 32:
            raise ValueError("Native adapter configuration differs")
        settings = {
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "max_model_len": 36864,
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 4096,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "language_model_only": True,
            "logprobs_mode": "processed_logprobs",
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": 16,
            "seed": 3407,
            "worker_extension_cls": "sampler_compat.NativeSamplingWorker",
            "lora_dtype": "bfloat16",
        }
        result.update(
            engine=settings,
            adapter_sha256=adapter_hash,
            adapter_target_modules=config["target_modules"],
            prompt_ids_sha256=request["prompt_ids_sha256"],
            model=request["model"],
            checkpoint=str(checkpoint),
        )
        mapped_checkpoint = args.out / "mapped-adapter"
        result["adapter_mapping"] = prepare_adapter(checkpoint, mapped_checkpoint)
        engine = LLM(model=str(args.model), tokenizer=str(args.model), **settings)
        scope.enter_context(native_engine_sampling(engine))
        result["sampler_filter"] = "native_torch"
        result["exact_adapter_precision_verified"] = False
        result["load_seconds"] = time.monotonic() - started
        prompt = {"prompt_token_ids": request["prompt_ids"]}
        adapter = LoRARequest("checkpoint-1-" + adapter_hash[:12], 1, str(mapped_checkpoint))
        probes = []
        for name, active in (("base", None), ("adapter", adapter), ("adapter-repeat", adapter)):
            before = time.monotonic()
            output = engine.generate(
                [prompt],
                SamplingParams(
                    temperature=0,
                    max_tokens=32,
                    logprobs=1,
                    seed=3407,
                ),
                lora_request=active,
                use_tqdm=False,
            )
            probe = capture_batch(
                output[0],
                request["prompt_ids"],
                count=1,
                seconds=time.monotonic() - before,
                logprob_semantics="sampling_processed",
                weight_version=(adapter_hash if active else "base:" + request["model"]["revision"]),
            )
            (args.out / f"probe-{name}.json").write_text(json.dumps(probe) + "\n")
            probes.append(probe)
        result["adapter_changes_outputs_or_logprobs"] = outputs_changed(probes[0], probes[1])
        result["repeated_adapter_stable"] = not outputs_changed(probes[1], probes[2])
        result["repeated_adapter_tokens_stable"] = repeated_token_ids(probes[1], probes[2])
        result["logprob_repeat_max_difference"] = max(
            abs(a - b)
            for a, b in zip(
                probes[1]["draws"][0]["logprobs"],
                probes[2]["draws"][0]["logprobs"],
                strict=True,
            )
        )
        result["full_model_logprob_parity_verified"] = False
        result["grpo_logprob_integration_approved"] = False
        if not (
            result["adapter_changes_outputs_or_logprobs"]
            and result["repeated_adapter_tokens_stable"]
        ):
            raise ValueError("Adapter effect or repeat stability check failed")
        result["status"] = "generating_full_budget_group"
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        before = time.monotonic()
        output = engine.generate(
            [prompt],
            SamplingParams(
                n=4,
                temperature=1,
                top_p=0.95,
                top_k=20,
                max_tokens=request["full_completion_ceiling"],
                logprobs=1,
                seed=3407,
            ),
            lora_request=adapter,
            use_tqdm=False,
        )
        batch = capture_batch(
            output[0],
            request["prompt_ids"],
            count=4,
            seconds=time.monotonic() - before,
            logprob_semantics="sampling_processed",
            weight_version=adapter_hash,
        )
        batch["logprobs_mode"] = "processed_logprobs"
        (args.out / "batch-4-full.json").write_text(json.dumps(batch) + "\n")
        result["batches"].append({key: value for key, value in batch.items() if key != "draws"})
        result["status"] = "adapter_and_full_budget_generation_completed"
    except Exception as exc:
        result.update(status="integration_check_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            scope.close()
        finally:
            result["total_seconds"] = time.monotonic() - started
            result["estimated_compute_usd"] = result["total_seconds"] / 3600 * 2.51
            (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
