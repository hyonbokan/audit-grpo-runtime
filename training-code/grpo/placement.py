"""Where the update and rollout put the model: the GPUs each phase sees and how layers split."""

from __future__ import annotations

from typing import Any

EMBEDDING_PLACEMENTS = ("offloaded", "gpu")
PLACEMENT_KEYS = {"gpus", "embedding", "max_memory_gib"}
# Both A6000s of this host; a larger tensor-parallel group needs more visible devices.
MAXIMUM_TENSOR_PARALLEL = 2


def load_options(placement: dict[str, Any]) -> dict[str, Any]:
    """`FastModel.from_pretrained` keywords that place the frozen base as reviewed."""
    gpus = validate_placement(placement)
    offload = placement["embedding"] == "offloaded"
    if len(gpus) == 1:
        # An explicit map stops Unsloth's default planner from splitting over every visible card.
        return {"device_map": {"": 0}, "offload_embedding": offload}
    cap = f"{placement['max_memory_gib']}GiB"
    # Inside the container the listed GPUs are renumbered from 0.
    return {
        "device_map": "unsloth",
        "max_memory": dict.fromkeys(range(len(gpus)), cap),
        "offload_embedding": False,
    }


def validate_placement(placement: dict[str, Any]) -> list[int]:
    unknown = set(placement) - PLACEMENT_KEYS
    if unknown:
        raise ValueError(f"Unknown update_placement keys: {sorted(unknown)}")
    gpus = placement["gpus"]
    if (
        not gpus
        or len(set(gpus)) != len(gpus)
        or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
    ):
        raise ValueError(f"update_placement.gpus must be distinct GPU indexes, got {gpus!r}")
    if placement["embedding"] not in EMBEDDING_PLACEMENTS:
        raise ValueError(f"update_placement.embedding must be one of {EMBEDDING_PLACEMENTS}")
    split = len(gpus) > 1
    if split and placement["embedding"] == "offloaded":
        raise ValueError("Unsloth declines embedding offload for a model split across GPUs")
    cap = placement.get("max_memory_gib")  # Only a split has a per-device cap.
    if split != (cap is not None) or (split and (type(cap) not in (int, float) or not cap > 0)):
        raise ValueError("update_placement.max_memory_gib is a positive cap, set only for a split")
    return gpus


def docker_gpus(gpus: list[int]) -> str:
    """The `docker run --gpus` value that exposes exactly these devices."""
    return '"device=' + ",".join(str(gpu) for gpu in gpus) + '"'


def rollout_gpus(engine: dict[str, Any]) -> list[int]:
    size = engine.get("tensor_parallel_size", 1)  # vLLM's own default: one GPU.
    if type(size) is not int or not 1 <= size <= MAXIMUM_TENSOR_PARALLEL:
        raise ValueError(f"engine.tensor_parallel_size must be 1..{MAXIMUM_TENSOR_PARALLEL}")
    return list(range(size))


def layer_devices(layers: list[Any]) -> dict[str, int]:
    """Decoder layers per device; block-swapped layers report their host copy."""
    counts: dict[str, int] = {}
    for index, layer in enumerate(layers):
        devices = {str(parameter.device) for parameter in layer.parameters()}
        if len(devices) != 1:
            raise ValueError(f"Decoder layer {index} spans devices {sorted(devices)}")
        (device,) = devices
        counts[device] = counts.get(device, 0) + 1
    return dict(sorted(counts.items()))
