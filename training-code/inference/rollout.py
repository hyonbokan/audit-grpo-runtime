"""Versioned four-draw training rollout or evaluation arms in a separate vLLM process."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import benchmark_free_text as free_text
from adapter_compat import verify_tensor_parallel
from adapter_mapping import prepare_adapter
from cycle_contract import validate_config, validate_policy_profile, write_event
from generation import capture_batch, validate_request, validate_shared_engine
from grpo.samples import capture_sample
from runtime_io import read_json, save_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    PHASES[args.phase](parser, args)
    return 0


def phase_single(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.config is None or args.out is None:
        parser.error(f"{args.phase} requires --config and --out")
    run_rollout(args.phase, read_json(args.config), args.out)


def phase_suite(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.manifest is None:
        parser.error("evaluate-suite requires --manifest")
    run_evaluation_suite(read_json(args.manifest))


PHASES = {"rollout": phase_single, "evaluate": phase_single, "evaluate-suite": phase_suite}


# --- One arm per engine


def run_rollout(phase: str, config: dict[str, Any], out: Path) -> dict[str, Any]:
    arm = resolve_arm(phase, config)
    out.mkdir(parents=True, exist_ok=False)
    receipt = arm_receipt(arm)
    started = time.monotonic()
    write_event(out, "start", weight_version=arm["version"])
    try:
        engine, adapter, sampling_params = load_engine(arm, out, receipt)
        settings = dict(config["sampling"])
        generation_start = time.monotonic()
        output = engine.generate(
            [{"prompt_token_ids": arm["request"]["prompt_ids"]}],
            sampling_params(**settings),
            use_tqdm=False,
            **({} if arm["merged"] else {"lora_request": adapter}),
        )
        seconds = time.monotonic() - generation_start
        batch = save_arm(engine, arm, output[0], settings, seconds, out)
        receipt.update(
            status="versioned_rollout_saved",
            completion_tokens=batch["completion_tokens"],
            generation_seconds=seconds,
            tokens_per_second=batch["aggregate_tokens_per_second"],
        )
        write_event(
            out, "end", tokens=receipt["completion_tokens"], seconds=receipt["generation_seconds"]
        )
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_event(out, "failure", reason=receipt["error"])
        raise
    finally:
        receipt["total_seconds"] = time.monotonic() - started
        save_json(out / "receipt.json", receipt)
    return receipt


def resolve_arm(phase: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate one arm's config, request and weights, and name the version it samples."""
    if phase == "evaluate":
        free_text.validate_config(config)
    else:
        validate_config(config)
    request = read_json(config["request"])
    validate_request(request)
    if config.get("policy_reasoning_effort"):
        validate_policy_profile(config, request, read_json(config["source_plan"]))
    if (
        request["model"]["id"] != config["model_id"]
        or request["model"]["revision"] != config["model_revision"]
    ):
        raise ValueError("Pinned model request differs")
    step = config["initial_step"]
    base = phase == "evaluate" and config.get("label") == "base"
    if config.get("merged_model_path") and phase != "evaluate":
        raise ValueError("Merged weights are evaluation-only")
    merged = bool(config.get("merged_model_path")) and not base
    checkpoint = None if base else Path(config["checkpoint"])
    digest = (
        hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
        if checkpoint is not None
        else None
    )
    if not base and digest != config["initial_adapter_sha256"]:
        raise ValueError("Initial source checkpoint identity differs")
    version = f"base:{config['model_revision']}" if base else f"step-{step}:{digest}"
    representation = "base_bf16" if base else "merged_bf16" if merged else "runtime_lora"
    provenance = {"weight_representation": representation}
    if merged:
        manifest_path = Path(config["merged_model_path"]) / "MERGE_MANIFEST.json"
        manifest = read_json(manifest_path)
        if (
            manifest.get("status") != "merged_and_verified"
            or manifest.get("base_revision") != config["model_revision"]
            or manifest.get("adapter_sha256") != digest
            or manifest.get("stored_weight_precision") != "bfloat16"
        ):
            raise ValueError("Merged artifact source identity or precision differs")
        provenance.update(
            merged_model_path=config["merged_model_path"],
            merge_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            numerical_agreement_verified=False,
        )
    return {
        "config": config,
        "request": request,
        "phase": phase,
        "step": step,
        "merged": merged,
        "checkpoint": checkpoint,
        "digest": digest,
        "version": version,
        "provenance": provenance,
        "runtime_adapter": checkpoint is not None and not merged,
    }


def arm_receipt(arm: dict[str, Any]) -> dict[str, Any]:
    config = arm["config"]
    return {
        "status": "loading",
        "sampler_backend": config["sampler_backend"],
        "weight_version": arm["version"],
        "native_adapter_precision": config["native_adapter_precision"],
        "rollout_adapter_precision": config["rollout_adapter_precision"],
        "policy_template_options": arm["request"].get("template_options"),
        **arm["provenance"],
    }


def load_engine(arm: dict[str, Any], out: Path, receipt: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Start vLLM on the arm's weights, verify a runtime adapter, and return its param type."""
    # vLLM and msgspec exist only in the rollout image.
    import msgspec
    from vllm import LLM, SamplingParams

    config, checkpoint, step, digest = arm["config"], arm["checkpoint"], arm["step"], arm["digest"]
    mapped = out / "mapped-adapter"
    if arm["runtime_adapter"]:
        adapter_config = read_json(checkpoint / "adapter_config.json")
        if adapter_config["r"] != config["engine"]["max_lora_rank"]:
            raise ValueError("Adapter rank differs from engine configuration")
        receipt["mapping"] = prepare_adapter(checkpoint, mapped)
    engine_options = dict(config["engine"])
    if arm["merged"]:
        for key in ("lora_dtype", "max_loras", "max_lora_rank"):
            engine_options.pop(key, None)
        engine_options["enable_lora"] = False
    elif engine_options.get("enable_lora"):
        engine_options["worker_extension_cls"] = "adapter_compat.VerifiedAdapterWorker"
    engine = LLM(
        model=config["merged_model_path"] if arm["merged"] else config["model_path"],
        tokenizer=config["model_path"],
        **engine_options,
    )
    if not arm["runtime_adapter"]:
        return engine, None, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter = LoRARequest(f"checkpoint-{step}-{digest[:12]}", step, str(mapped), load_inplace=True)
    engine.collective_rpc("load_adapter", args=(msgspec.to_builtins(adapter),))
    receipt["transfer"] = verify_active_adapter(engine, config, step, checkpoint, digest, out)
    write_event(
        out,
        "active_adapter_verified",
        weight_version=arm["version"],
        tensors=receipt["transfer"]["source_tensor_count"],
    )
    return engine, adapter, SamplingParams


def save_arm(
    engine: Any,
    arm: dict[str, Any],
    output: Any,
    settings: dict[str, Any],
    seconds: float,
    out: Path,
    shared_arms: int | None = None,
) -> dict[str, Any]:
    """Capture the engine's draws for one arm and write its samples and batch record."""
    request, version = arm["request"], arm["version"]
    batch = capture_batch(
        output,
        request["prompt_ids"],
        count=settings["n"],
        seconds=seconds,
        logprob_semantics="sampling_processed",
        weight_version=version,
        require_sampling_support=arm["phase"] != "evaluate",
    )
    # Capture has complete token IDs; decode through the actual pinned tokenizer.
    tokenizer = engine.get_tokenizer()
    samples = []
    for draw in batch["draws"]:
        sample = capture_sample(
            request["row_id"],
            request["prompt_ids"],
            draw["completion_ids"],
            draw["logprobs"],
            tokenizer,
            finish_reason=draw["finish_reason"],
        )
        sample.update(
            logprob_semantics="sampling_processed",
            weight_version=version,
            raw_logprobs=None,
            sampling_support_ids=draw["sampling_support_ids"],
            **arm["provenance"],
        )
        samples.append(sample)
    (out / "samples.json").write_text(json.dumps(samples) + "\n")
    # A shared call's duration covers every arm, so per-arm rates are not comparable.
    record = batch if shared_arms is None else dict(batch, shared_generation_arms=shared_arms)
    (out / "batch.json").write_text(json.dumps(record) + "\n")
    return batch


def verify_active_adapter(
    engine: Any, config: dict[str, Any], step: int, checkpoint: Path, digest: str, out: Path
) -> dict[str, Any]:
    """Every rank's active LoRA buffers equal the native adapter."""
    size = config["engine"].get("tensor_parallel_size", 1)  # vLLM's own default: one GPU.
    if size == 1:
        reports = engine.collective_rpc("verify_adapter", args=(step, str(checkpoint), digest))
        if len(reports) != 1:
            raise ValueError("Exactly one local worker receipt required")
        return reports[0]
    # Each rank writes its buffers here; they are deleted once verified.
    directory = out / "adapter-shards"
    try:
        replies = engine.collective_rpc("adapter_shards", args=(step, str(directory)))
        return verify_tensor_parallel(replies, checkpoint, digest, size=size)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# --- Every project's arm in one engine


def run_evaluation_suite(manifest: dict[str, Any]) -> dict[str, Any]:
    """Load the weights once and sample every project's arm in one call to the engine."""
    arms = [
        (resolve_arm("evaluate", read_json(entry["config"])), Path(entry["out"]))
        for entry in manifest["arms"]
    ]
    validate_suite_arms([arm for arm, _ in arms])
    receipt_path = Path(manifest["receipt"])
    first = arms[0][0]
    suite = {
        "status": "loading",
        "weight_version": first["version"],
        "arms": [str(out) for _, out in arms],
        **first["provenance"],
    }
    receipts = {}
    for arm, out in arms:
        out.mkdir(parents=True, exist_ok=False)
        # The shared call has one duration, recorded once in the suite receipt.
        receipts[out] = dict(arm_receipt(arm), shared_generation_receipt=str(receipt_path))
        write_event(out, "start", weight_version=arm["version"])
    started = time.monotonic()
    try:
        engine, _, sampling_params = load_engine(first, arms[0][1], suite)
        settings = [dict(arm["config"]["sampling"]) for arm, _ in arms]
        generation_start = time.monotonic()
        outputs = engine.generate(
            [{"prompt_token_ids": arm["request"]["prompt_ids"]} for arm, _ in arms],
            [sampling_params(**values) for values in settings],
            use_tqdm=False,
        )
        seconds = time.monotonic() - generation_start
        if len(outputs) != len(arms):
            raise ValueError("Engine returned a different number of requests")
        tokens = 0
        for (arm, out), output, values in zip(arms, outputs, settings, strict=True):
            batch = save_arm(engine, arm, output, values, seconds, out, shared_arms=len(arms))
            receipts[out].update(
                status="versioned_rollout_saved", completion_tokens=batch["completion_tokens"]
            )
            tokens += batch["completion_tokens"]
            write_event(out, "end", tokens=batch["completion_tokens"])
        suite.update(
            status="versioned_rollout_saved",
            completion_tokens=tokens,
            generation_seconds=seconds,
            tokens_per_second=tokens / seconds,
        )
    except Exception as error:
        suite.update(status="failed", error=f"{type(error).__name__}: {error}")
        for _, out in arms:
            if receipts[out]["status"] != "versioned_rollout_saved":
                receipts[out].update(status="failed", error=suite["error"])
                write_event(out, "failure", reason=suite["error"])
        raise
    finally:
        suite["total_seconds"] = time.monotonic() - started
        for out, receipt in receipts.items():
            save_json(out / "receipt.json", receipt)
        save_json(receipt_path, suite)
    return suite


def validate_suite_arms(arms: list[dict[str, Any]]) -> None:
    """One engine serves every arm: the same base or merged weights and engine settings."""
    validate_shared_engine([arm["config"] for arm in arms])
    for arm in arms:
        if arm["runtime_adapter"]:
            raise ValueError("A shared engine serves base or merged weights, not runtime adapters")
        if arm["version"] != arms[0]["version"]:
            raise ValueError("Shared engine arms differ in weight version")


if __name__ == "__main__":
    raise SystemExit(main())
