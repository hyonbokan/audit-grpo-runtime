"""Verify every native LoRA tensor in the loaded and active vLLM adapter."""

import math
from typing import Any

import torch


def verify_transfer(
    source: dict[str, torch.Tensor], payload: dict[str, Any], *, scaling: float
) -> dict[str, Any]:
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("Invalid configured adapter scaling")
    failures = []
    verified = 0
    max_difference = 0.0
    cast_failures = []
    packed = {
        part: (name, index)
        for name, parts in payload["packed"].items()
        for index, part in enumerate(parts)
    }
    for key, tensor in source.items():
        module, label, _ = key.removeprefix("base_model.model.").rsplit(".", 2)
        module = module.replace("model.layers.", "language_model.model.layers.", 1)
        kind = "lora_a" if label == "lora_A" else "lora_b"
        module, index = packed.get(module, (module, 0))
        expected = tensor if kind == "lora_a" else tensor * scaling
        ok = True
        for storage, suffix in (("cpu", kind), ("gpu", kind + "_stacked")):
            values = payload[storage].get(module + "." + suffix, [])
            if storage == "gpu" and module.endswith(".in_proj_qkvz") and len(values) == 4:
                if kind == "lora_a":
                    if not all(torch.equal(values[0], value) for value in values[1:3]):
                        failures.append({"key": key, "reason": "GDN shared A slices differ"})
                        cast_failures.append(key)
                        ok = False
                    values = [values[0], values[3]]
                else:
                    values = [torch.cat([value[0] for value in values[:3]]), values[3]]
            if module not in payload["modules"] or index >= len(values) or values[index] is None:
                failures.append({"key": key, "storage": storage, "reason": "missing active tensor"})
                ok = False
                cast_failures.append(key)
                continue
            actual = values[index]
            if actual.ndim == 3 and actual.shape[0] == 1:
                actual = actual[0]
            if actual.ndim != 2 or any(
                a < b for a, b in zip(actual.shape, expected.shape, strict=False)
            ):
                failures.append({"key": key, "storage": storage, "reason": "shape mismatch"})
                ok = False
                cast_failures.append(key)
                continue
            actual = actual[: expected.shape[0], : expected.shape[1]]
            cast_expected = tensor.to(actual.dtype)
            if kind == "lora_b":
                cast_expected = cast_expected * scaling
            if not actual.equal(cast_expected):
                cast_failures.append(key)
            actual = actual.float()
            difference = (actual - expected.float()).abs().max().item()
            max_difference = max(max_difference, difference)
            if difference != 0:
                failures.append(
                    {
                        "key": key,
                        "storage": storage,
                        "reason": "value mismatch",
                        "max_abs_difference": difference,
                    }
                )
                ok = False
        verified += int(ok)
    return {
        "exact_transfer_verified": bool(source) and not failures,
        "cast_transfer_verified": bool(source) and not cast_failures,
        "cast_failure_keys": sorted(set(cast_failures)),
        "source_tensor_count": len(source),
        "verified_tensor_count": verified,
        "max_abs_source_difference": max_difference,
        "failures": failures,
        "lora_B_scaling_verified": scaling,
    }


def merge_tensor_parallel(shards: list[dict[str, Any]], *, size: int) -> dict[str, Any]:
    """One full-layout payload from each rank's buffers, for `verify_transfer`."""
    ordered = sorted(shards, key=lambda shard: shard["rank"])
    if [shard["rank"] for shard in ordered] != list(range(size)):
        raise ValueError(f"Adapter verification needs one shard per rank 0..{size - 1}")
    first = ordered[0]
    layout = ("packed", "modules", "sharded_dims")
    for shard in ordered[1:]:
        if any(shard[key] != first[key] for key in layout) or set(shard["gpu"]) != set(
            first["gpu"]
        ):
            raise ValueError(f"Adapter layout differs between rank 0 and rank {shard['rank']}")
    gpu = {}
    for key, slices in first["gpu"].items():
        dim = first["sharded_dims"].get(key)  # Absent: vLLM replicates this buffer.
        merged = []
        for index in range(len(slices)):
            parts = [shard["gpu"][key][index] for shard in ordered]
            if dim is not None:
                merged.append(torch.cat(parts, dim=dim))
            elif all(torch.equal(parts[0], part) for part in parts[1:]):
                merged.append(parts[0])
            else:
                raise ValueError(f"Replicated adapter buffer differs between ranks: {key}")
        gpu[key] = merged
    # The loaded CPU copy is the full unsliced adapter on every rank; rank 0 reports it.
    return {"packed": first["packed"], "modules": first["modules"], "cpu": first["cpu"], "gpu": gpu}
