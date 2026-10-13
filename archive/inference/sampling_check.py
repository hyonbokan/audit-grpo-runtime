"""Bounded acceptance of the maintained sampler change using retained GPU evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acceptance import IntegrationEvidence, evaluate_integration
from benchmark import validate_request
from sampler_compat import native_sampling_scope


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate_limits(config)
    result = run_sampling_check(config, args.out)
    print(
        json.dumps(
            {
                "sampler_implementation_accepted": result["sampler_implementation_accepted"],
                "grpo_decision": result["grpo_decision"],
            }
        )
    )
    return 0 if result["sampler_implementation_accepted"] else 1


def run_sampling_check(config: dict[str, Any], out: Path) -> dict[str, Any]:
    validate_limits(config)
    paths = {
        name: Path(config[name])
        for name in (
            "request",
            "logits",
            "expected",
            "prior_analysis",
            "step_receipt",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    deadline = datetime.fromisoformat(config["absolute_deadline"])
    if (deadline - datetime.now(timezone.utc)).total_seconds() < config["gpu_seconds_limit"]:
        raise ValueError("Insufficient remaining approved time budget")
    request = json.loads(paths["request"].read_text())
    validate_request(request)
    prior = json.loads(paths["prior_analysis"].read_text())
    step = json.loads(paths["step_receipt"].read_text())
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "sampler_implementation_accepted": False,
        "optimizer_updates": 0,
        "judge_calls": 0,
        "question": "Does maintained V2 filtering preserve recorded native CUDA probabilities?",
        "scope": "same-logits sampler implementation; no model load or training",
        "model": request["model"],
        "sampling": request["sampling"],
    }
    (out / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    result["evidence_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    }
    source_root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }
    record_event(out, "start", gpu_seconds_limit=config["gpu_seconds_limit"])
    try:
        # CUDA/V2 imports belong to the explicit GPU application, not CPU preparation.
        import numpy as np
        import torch
        from vllm import SamplingParams
        from vllm.v1.worker.gpu.sample import sampler, states
        from vllm.v1.worker.gpu.sample.states import SamplingStates

        torch.cuda.set_device(0)
        logits = torch.load(paths["logits"], weights_only=True).cuda()
        expected = torch.load(paths["expected"], weights_only=True)
        if logits.shape != expected.shape:
            raise ValueError("Recorded logits/probabilities have different shapes")
        sampling = SamplingStates(logits.shape[0], logits.shape[1])
        for index in range(logits.shape[0]):
            sampling.add_request(index, SamplingParams(**request["sampling"]))
        sampling.apply_staged_writes()
        indices = torch.arange(logits.shape[0], device="cuda")
        original_sampler, original_states = sampler.apply_top_k_top_p, states.apply_top_k_top_p
        with native_sampling_scope():
            masked = sampling.apply_top_k_top_p(logits, indices, np.arange(logits.shape[0]))
            actual = masked.log_softmax(-1).cpu()
        restored = (
            sampler.apply_top_k_top_p is original_sampler
            and states.apply_top_k_top_p is original_states
        )
        support_equal = torch.equal(actual.isfinite(), expected.isfinite())
        common = actual.isfinite() & expected.isfinite()
        difference = (actual[common] - expected[common]).abs().max().item()
        accepted = support_equal and difference <= config["atol"] and restored
        result.update(
            sampler_implementation_accepted=accepted,
            support_equal=support_equal,
            max_common_logprob_difference=difference,
            hooks_restored=restored,
            support_sizes=actual.isfinite().sum(-1).tolist(),
            atol=config["atol"],
        )
        evidence = IntegrationEvidence(
            sampling_semantics=accepted,
            raw_probability_parity=prior["matched_contexts"]["raw_probability_parity"],
            processed_probability_parity=prior["matched_contexts"]["processed_probability_parity"],
            repeat_stability=(
                prior["repeat"]["raw_probability_parity"]
                and prior["repeat"]["processed_probability_parity"]
            ),
            adapter_transfer=prior["weight_transfer"]["cast_transfer_verified"],
            adapter_precision=prior["weight_transfer"]["exact_transfer_verified"],
            trainer_refresh=prior["automatic_live_trainer_refresh_verified"],
            changed_weight_progression=step["adapter_weights_changed"],
        )
        result["grpo_decision"] = asdict(evaluate_integration(evidence))
        result["grpo_evidence"] = asdict(evidence)
        torch.save(actual, out / "maintained-v2-processed.pt")
        record_event(out, "end", accepted=accepted)
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", phase="sampling_acceptance")
        record_event(out, "failure", reason=result["error"])
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        result["estimated_job_compute_usd"] = result["seconds"] / 3600 * config["rate_usd_per_hour"]
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def validate_limits(config: dict[str, Any]) -> None:
    seconds = config.get("gpu_seconds_limit")
    if type(seconds) is not int or not 0 < seconds <= 120:
        raise ValueError("GPU time budget must be positive and at most 120 seconds")
    if config.get("retry_limit") != 0:
        raise ValueError("This acceptance run has a zero retry budget")
    if config.get("atol") != 1e-6:
        raise ValueError("The reviewed same-logits sampler tolerance is fixed at 1e-6")


def record_event(out: Path, event: str, **details: Any) -> None:
    value = {
        "phase": "sampling_acceptance",
        "event": event,
        "utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    with (out / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
