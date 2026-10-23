"""Create the step-0 checkpoint: an adapter equal to the base model and an empty optimizer."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from runtime_io import save_json, sha256_file

COPIED_FILES = ("adapter_config.json",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.template, args.out, args.seed, args.learning_rate), indent=2))
    return 0


def create(template: Path, out: Path, seed: int, learning_rate: float) -> dict[str, Any]:
    """A resumable step-0 checkpoint shaped like `template`, a checkpoint of the same model."""
    if out.exists():
        raise FileExistsError(out)
    generator = torch.Generator().manual_seed(seed)
    tensors = load_file(template / "adapter_model.safetensors")
    fresh = {name: fresh_tensor(name, tensors[name], generator) for name in sorted(tensors)}
    optimizer = torch.load(template / "optimizer.pt", map_location="cpu", weights_only=True)
    if sum(len(group["params"]) for group in optimizer["param_groups"]) != len(fresh):
        raise ValueError("Template optimizer and adapter differ in parameter count")
    scheduler = torch.load(template / "scheduler.pt", map_location="cpu", weights_only=True)
    out.mkdir(parents=True)
    save_file(fresh, out / "adapter_model.safetensors", metadata={"format": "pt"})
    for name in COPIED_FILES:
        (out / name).write_bytes((template / name).read_bytes())
    torch.save({**optimizer, "state": {}}, out / "optimizer.pt")
    # The scheduler keeps one rate per optimizer group, empty groups included.
    rates = [learning_rate] * len(optimizer["param_groups"])
    scheduler.update(last_epoch=0, _step_count=1, base_lrs=rates, _last_lr=list(rates))
    torch.save(scheduler, out / "scheduler.pt")
    torch.save(rng_state(seed), out / "rng_state.pth")
    save_json(out / "trainer_state.json", {"global_step": 0})
    schedule = {"kind": "initial", "initial_step": 0, "seed": seed, "learning_rate": learning_rate}
    save_json(out / "continuation_schedule.json", schedule)
    return {
        "checkpoint": str(out),
        "adapter_sha256": sha256_file(out / "adapter_model.safetensors"),
    }


def fresh_tensor(name: str, template: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """PEFT's default start: B zero, so nothing changes; A uniform in ±1/sqrt(fan_in)."""
    if name.endswith("lora_B.weight"):
        return torch.zeros_like(template)
    if name.endswith("lora_A.weight"):
        bound = 1 / math.sqrt(template.shape[1])
        return torch.empty_like(template).uniform_(-bound, bound, generator=generator)
    raise ValueError(f"Adapter tensor {name!r} is neither lora_A nor lora_B")


def rng_state(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Made on a CPU host: the update seeds each GPU from its own default.
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
