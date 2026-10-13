"""Versioned four-draw rollout or post-update refresh in a separate vLLM process."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from adapter_mapping import prepare_adapter
from cycle_contract import validate_config, validate_policy_profile, write_event
from evaluation import validate_evaluation_config
from generation import capture_batch, validate_request
from grpo.samples import capture_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("rollout", "refresh", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_rollout(args.phase, json.loads(args.config.read_text()), args.out)
    return 0


def run_rollout(phase: str, config: dict[str, Any], out: Path) -> dict[str, Any]:
    if phase == "evaluate":
        validate_evaluation_config(config)
    else:
        validate_config(config)
    request = json.loads(Path(config["request"]).read_text())
    validate_request(request)
    if config.get("policy_reasoning_effort"):
        validate_policy_profile(
            config, request, json.loads(Path(config["source_plan"]).read_text())
        )
    if (
        request["model"]["id"] != config["model_id"]
        or request["model"]["revision"] != config["model_revision"]
    ):
        raise ValueError("Pinned model request differs")
    step = config["target_step"] if phase == "refresh" else config["initial_step"]
    base = phase == "evaluate" and config.get("label") == "base"
    if config.get("merged_model_path") and phase != "evaluate":
        raise ValueError("Merged weights are evaluation-only")
    merged = bool(config.get("merged_model_path")) and not base
    checkpoint = (
        None
        if base
        else (
            Path(config["checkpoint"])
            if phase != "refresh"
            else out.parent / f"update/checkpoint-{step}"
        )
    )
    digest = (
        hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
        if checkpoint is not None
        else None
    )
    if not base and phase != "refresh" and digest != config["initial_adapter_sha256"]:
        raise ValueError("Initial source checkpoint identity differs")
    if phase == "refresh":
        updated = json.loads((out.parent / "update/receipt.json").read_text())
        if (
            updated["adapter_sha256"] != digest
            or updated["global_step"] != step
            or digest == config["initial_adapter_sha256"]
        ):
            raise ValueError("Fresh trainer checkpoint identity/progression differs")
    version = f"base:{config['model_revision']}" if base else f"step-{step}:{digest}"
    representation = "base_bf16" if base else "merged_bf16" if merged else "runtime_lora"
    provenance = {"weight_representation": representation}
    if merged:
        manifest_path = Path(config["merged_model_path"]) / "MERGE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
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
    out.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "status": "loading",
        "sampler_backend": config["sampler_backend"],
        "weight_version": version,
        "native_adapter_precision": config["native_adapter_precision"],
        "rollout_adapter_precision": config["rollout_adapter_precision"],
        "policy_template_options": request.get("template_options"),
        **provenance,
    }
    started = time.monotonic()
    write_event(out, "start", weight_version=version)
    try:
        import msgspec
        from vllm import LLM, SamplingParams

        mapped = out / "mapped-adapter"
        if checkpoint is not None and not merged:
            adapter_config = json.loads((checkpoint / "adapter_config.json").read_text())
            if adapter_config["r"] != config["engine"]["max_lora_rank"]:
                raise ValueError("Adapter rank differs from engine configuration")
            receipt["mapping"] = prepare_adapter(checkpoint, mapped)
        engine_options = dict(config["engine"])
        if merged:
            for key in ("lora_dtype", "max_loras", "max_lora_rank"):
                engine_options.pop(key, None)
            engine_options["enable_lora"] = False
        elif engine_options.get("enable_lora"):
            engine_options["worker_extension_cls"] = "adapter_compat.VerifiedAdapterWorker"
        engine = LLM(
            model=config["merged_model_path"] if merged else config["model_path"],
            tokenizer=config["model_path"],
            **engine_options,
        )
        adapter = None
        if checkpoint is not None and not merged:
            from vllm.lora.request import LoRARequest

            adapter = LoRARequest(
                f"checkpoint-{step}-{digest[:12]}", step, str(mapped), load_inplace=True
            )
            engine.collective_rpc("load_adapter", args=(msgspec.to_builtins(adapter),))
            reports = engine.collective_rpc("verify_adapter", args=(step, str(checkpoint), digest))
            if len(reports) != 1:
                raise ValueError("Exactly one local worker receipt required")
            receipt["transfer"] = reports[0]
            write_event(
                out,
                "active_adapter_verified",
                weight_version=version,
                tensors=reports[0]["source_tensor_count"],
            )
        settings = dict(config["sampling"])
        if phase == "refresh":
            settings.update(n=1, max_tokens=1)
        generation_start = time.monotonic()
        output = engine.generate(
            [{"prompt_token_ids": request["prompt_ids"]}],
            SamplingParams(**settings),
            use_tqdm=False,
            **({} if merged else {"lora_request": adapter}),
        )
        seconds = time.monotonic() - generation_start
        batch = capture_batch(
            output[0],
            request["prompt_ids"],
            count=settings["n"],
            seconds=seconds,
            logprob_semantics="sampling_processed",
            weight_version=version,
            require_sampling_support=phase != "evaluate",
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
                **provenance,
            )
            samples.append(sample)
        (out / "samples.json").write_text(json.dumps(samples) + "\n")
        (out / "batch.json").write_text(json.dumps(batch) + "\n")
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
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
