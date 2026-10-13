"""One explicitly scheduled, mask-aware corrected update of a native policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from grpo.corrected_loss import corrected_loss, support_logprobs


def join_update_group(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    rewards: list[float],
    expected_version: str,
) -> list[dict[str, Any]]:
    if len(samples) != 4 or len(rewards) != 4:
        raise ValueError("Complete four-draw grades required")
    if any(
        sample.get("weight_version") != expected_version
        or (config.get("row_id") and sample.get("row_id") != config["row_id"])
        for sample in samples
    ):
        raise ValueError("Scheduled row or source weight differs")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in rewards):
        raise ValueError("Four finite known grades required")
    return [{**sample, "reward": reward} for sample, reward in zip(samples, rewards, strict=True)]


def apply_update(
    model: Any,
    optimizer: Any,
    group: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    expected_version: str,
) -> dict[str, Any]:
    import torch

    if len(group) != 4 or not expected_version:
        raise ValueError("Exactly four versioned draws required")
    for sample in group:
        if (
            sample.get("weight_version") != expected_version
            or sample.get("logprob_semantics") != "sampling_processed"
            or sample.get("sampling_support_ids") is None
            or not sample["completion_ids"]
            or len(sample["completion_ids"]) != len(sample["logprobs"])
            or sample["prompt_ids"] != group[0]["prompt_ids"]
            or sample["row_id"] != group[0]["row_id"]
            or type(sample.get("reward")) not in (int, float)
            or not math.isfinite(sample["reward"])
        ):
            raise ValueError("Incomplete, stale or inconsistent corrected group")
    rewards = [sample["reward"] for sample in group]
    if len(set(rewards)) < 2:
        raise ValueError("Group has no reward contrast")
    if any(g["lr"] != settings["learning_rate"] for g in optimizer.param_groups):
        raise ValueError("Continuation learning rate differs from reviewed schedule")
    model.train()
    decoder = model.get_decoder()
    head = model.get_output_embeddings()
    total_tokens = sum(len(sample["completion_ids"]) for sample in group)
    mean_reward = sum(rewards) / 4
    optimizer.zero_grad(set_to_none=True)
    loss_sum, ratios, old_logprobs = 0.0, [], []
    for sample in group:
        ids = sample["prompt_ids"] + sample["completion_ids"][:-1]
        # Native embedding/offload hooks retain placement; the decoder input is CUDA.
        device = head.weight.device
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        output = decoder(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state[0, len(sample["prompt_ids"]) - 1 :]
        current = support_logprobs(
            hidden,
            head.weight,
            sample["completion_ids"],
            sample["sampling_support_ids"],
            chunk=settings["projection_chunk"],
        )
        sampled = torch.tensor(sample["logprobs"], dtype=current.dtype, device=current.device)
        old = current.detach()
        loss, _ = corrected_loss(
            current,
            old,
            sampled,
            advantage=sample["reward"] - mean_reward,
            total_tokens=total_tokens,
            epsilon=settings["epsilon"],
            correction_max=settings["correction_max"],
        )
        ratios.append((old.double() - sampled.double()).exp().cpu())
        old_logprobs.append(old.cpu().tolist())
        loss_sum += loss.detach().item()
        loss.backward()
        del output, hidden, current, old, loss
    ratio = torch.cat(ratios)
    if not torch.isfinite(ratio).all().item():
        raise ValueError("Nonfinite untruncated correction ratio")
    clip_fraction = (ratio > settings["correction_max"]).double().mean().item()
    ess_fraction = ((ratio.sum() ** 2) / (ratio.square().sum() * ratio.numel())).item()
    if clip_fraction > settings["max_clip_fraction"] or ess_fraction < settings["min_ess_fraction"]:
        raise ValueError(
            f"Correction health rejected: clipping={clip_fraction}, ESS={ess_fraction}"
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, settings["max_grad_norm"], error_if_nonfinite=True
    )
    if not torch.isfinite(norm).item() or norm.item() <= 0:
        raise ValueError("A nonzero finite gradient is required")
    optimizer.step()
    return {
        "optimizer_updates": 1,
        "loss": loss_sum,
        "gradient_norm": norm.item(),
        "correction_tokens": total_tokens,
        "correction_clip_fraction": clip_fraction,
        "correction_ess_fraction": ess_fraction,
        "correction_min": ratio.min().item(),
        "correction_max_observed": ratio.max().item(),
        "native_old_logprobs": old_logprobs,
        "advantages": [reward - mean_reward for reward in rewards],
        "source_weight_version": expected_version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_update(json.loads(args.config.read_text()), args.out)
    return 0


def run_update(config: dict[str, Any], out: Path) -> dict[str, Any]:
    from cycle_contract import validate_config, validate_policy_profile, write_event

    validate_config(config)
    out.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {"status": "loading", "optimizer_updates": 0}
    try:
        import torch
        from safetensors.torch import load_file

        from grpo.native_policy import (
            load_policy,
            restore_optimizer,
            restore_scheduler,
            save_checkpoint,
        )

        from .memory_compat import install_cpu_checkpoints, restore_existing_embedding_offload

        checkpoint = Path(config["checkpoint"])
        plan = json.loads(Path(config["source_plan"]).read_text())
        if config.get("policy_reasoning_effort"):
            validate_policy_profile(config, json.loads(Path(config["request"]).read_text()), plan)
            receipt["policy_reasoning_effort"] = config["policy_reasoning_effort"]
        source_sha = hashlib.sha256(
            (checkpoint / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        expected_version = f"step-{config['initial_step']}:{source_sha}"
        samples = json.loads((out.parent / "rollout/samples.json").read_text())
        grades = json.loads((out.parent / "grading/rewards.json").read_text())
        group = join_update_group(config, samples, grades, expected_version)
        model, memory, layers = load_policy(plan, checkpoint, config)
        optimizer = restore_optimizer(
            model,
            checkpoint,
            learning_rate=config["learning_rate"],
            optimizer_kind=config["optimizer_kind"],
        )
        scheduler = (
            restore_scheduler(optimizer, checkpoint, learning_rate=config["learning_rate"])
            if config.get("schedule_id")
            else None
        )
        before = load_file(str(checkpoint / "adapter_model.safetensors"))
        receipt["embedding_bytes_offloaded"] = restore_existing_embedding_offload(model)
        receipt["checkpoint_layers"] = install_cpu_checkpoints(layers, memory)
        write_event(out, "optimizer_step_start", source_version=expected_version)
        metrics = apply_update(model, optimizer, group, config, expected_version=expected_version)
        if scheduler is not None:
            scheduler.step()
        receipt.update(metrics)
        receipt.update(
            save_checkpoint(
                model,
                optimizer,
                before,
                out / f"checkpoint-{config['target_step']}",
                config,
                scheduler=scheduler,
            )
        )
        receipt.update(
            status="corrected_update_saved",
            global_step=config["target_step"],
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            checkpoint_input_offloaded_bytes=memory.offloaded_bytes,
        )
        write_event(
            out, "checkpoint_save", step=config["target_step"], version=receipt["weight_version"]
        )
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_event(out, "failure", reason=receipt["error"])
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
