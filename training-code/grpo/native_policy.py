"""Pinned native loading and explicit checkpoint-one continuation state."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
from typing import Any


def restore_optimizer(
    model: Any,
    checkpoint: Path,
    *,
    learning_rate: float,
    optimizer_kind: str = "adamw_torch",
) -> Any:
    import torch

    saved = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(len(group["params"]) for group in saved["param_groups"]) != len(parameters):
        raise ValueError("Checkpoint optimizer parameter count differs")
    groups, position = [], 0
    for group in saved["param_groups"]:
        count = len(group["params"])
        current = parameters[position : position + count]
        groups.append({**group, "params": current, "lr": learning_rate})
        position += count
    if optimizer_kind == "adamw_8bit":
        from bitsandbytes.optim import AdamW8bit

        optimizer = AdamW8bit(groups)
        moment_keys = ("state1", "state2")
    elif optimizer_kind == "adamw_torch":
        optimizer = torch.optim.AdamW(groups)
        moment_keys = ("exp_avg", "exp_avg_sq")
    else:
        raise ValueError(f"Unsupported checkpoint optimizer: {optimizer_kind}")
    # The original SDK unwraps its own quantization state and preserves uint8
    # moments/maps. Validate after that public loader, before taking any update.
    optimizer.load_state_dict(saved)
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if any(
            state.get(key) is None or state[key].shape != parameter.shape for key in moment_keys
        ):
            raise ValueError("Checkpoint optimizer moment shape differs")
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = learning_rate
    return optimizer


def restore_scheduler(optimizer: Any, checkpoint: Path, *, learning_rate: float) -> Any:
    import torch

    saved = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    scheduler.load_state_dict(saved)
    if any(lr != learning_rate for lr in scheduler.base_lrs + scheduler.get_last_lr()):
        raise ValueError("Saved scheduler differs from reviewed constant learning rate")
    return scheduler


def load_policy(
    plan: dict[str, Any], checkpoint: Path, config: dict[str, Any]
) -> tuple[Any, Any, list[Any]]:
    # Unsloth must patch model imports before Transformers/PEFT are imported.
    import unsloth  # noqa: F401, I001
    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from unsloth import FastModel

    from .memory_compat import ActivationOffload, configure_frozen_layer_offload

    actual = {name: importlib.metadata.version(name) for name in plan["package_versions"]}
    if actual != plan["package_versions"]:
        raise ValueError("Native package pins differ")
    adapter_config = json.loads((checkpoint / "adapter_config.json").read_text())
    model, _ = FastModel.from_pretrained(
        model_name=plan["model"]["id"],
        revision=plan["model"]["revision"],
        use_exact_model_name=True,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        fast_inference=False,
        text_only=True,
        return_logits=True,
        max_seq_length=plan["model"]["context_limit"],
        offload_embedding=True,
    )
    model = FastModel.get_peft_model(
        model,
        r=adapter_config["r"],
        lora_alpha=adapter_config["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
        target_modules=adapter_config["target_modules"],
    )
    loaded = set_peft_model_state_dict(
        model, load_file(str(checkpoint / "adapter_model.safetensors"))
    )
    if loaded.unexpected_keys:
        raise ValueError("Unexpected native adapter keys")
    if any(
        parameter.dtype != torch.float32
        for parameter in model.parameters()
        if parameter.requires_grad
    ):
        raise ValueError("Native trainable adapter precision must remain FP32")
    configure_frozen_layer_offload(
        model, layers=config["offload_layers"], prefetch_depth=config["prefetch_depth"]
    )
    FastModel.for_training(model, use_gradient_checkpointing="unsloth")
    model.enable_input_require_grads()
    model.train()
    layers = [
        module for module in model.modules() if type(module).__name__ == "Qwen3_5DecoderLayer"
    ]
    if len(layers) != 64:
        raise ValueError("Pinned native decoder layout differs")
    rng = torch.load(checkpoint / "rng_state.pth", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["cpu"])
    if isinstance(rng["cuda"], list):
        torch.cuda.set_rng_state_all(rng["cuda"])
    else:
        torch.cuda.set_rng_state(rng["cuda"])
    torch.cuda.reset_peak_memory_stats()
    return model, ActivationOffload(model), layers


def save_checkpoint(
    model: Any,
    optimizer: Any,
    before: dict[str, Any],
    destination: Path,
    config: dict[str, Any],
    *,
    scheduler: Any = None,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict

    after = get_peft_model_state_dict(model)
    changed = set(before) == set(after) and any(
        not torch.equal(before[name], value.detach().cpu()) for name, value in after.items()
    )
    if not changed:
        raise ValueError("Corrected update did not change adapter weights")
    steps = sorted({int(value["step"]) for value in optimizer.state.values()})
    if steps != [config["target_step"]]:
        raise ValueError(f"Optimizer progression differs: {steps}")
    if destination.exists():
        raise FileExistsError(destination)
    # Write beside the destination and rename once complete, so an interrupted save
    # never looks like a usable checkpoint and a resumed update can start cleanly.
    final, destination = destination, destination.with_name(destination.name + ".partial")
    destination.mkdir(exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    torch.save(scheduler.state_dict(), destination / "scheduler.pt")
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        destination / "rng_state.pth",
    )
    schedule = {
        "kind": "constant_single_update",
        "initial_step": config["initial_step"],
        "target_step": config["target_step"],
        "learning_rate": config["learning_rate"],
        "warmup_steps": 0,
        "updates_remaining": 0,
    }
    if config.get("continuation_schedule"):
        from schedules import schedule_after_update

        schedule = schedule_after_update(config)
    (destination / "continuation_schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    (destination / "trainer_state.json").write_text(
        json.dumps({"global_step": config["target_step"]}) + "\n"
    )
    destination.rename(final)
    destination = final
    digest = hashlib.sha256((destination / "adapter_model.safetensors").read_bytes()).hexdigest()
    return {
        "adapter_weights_changed": True,
        "optimizer_steps": steps,
        "adapter_sha256": digest,
        "weight_version": f"step-{config['target_step']}:{digest}",
        "checkpoint": str(destination),
        "continuation_schedule": schedule,
    }
