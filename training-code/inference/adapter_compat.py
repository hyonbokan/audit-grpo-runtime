"""Pinned worker-buffer inspection; private LoRA layout access stays here."""

import hashlib
import json
from pathlib import Path
from typing import Any

from sampler_compat import NativeSamplingWorker, require_pinned_vllm
from transfer import merge_tensor_parallel, verify_transfer

# The buffer dimension vLLM 0.31 slices per rank in `set_lora`, by LoRA layer class. Packed
# slices each hold their own rows on every rank, so rank order rebuilds them; absent buffers
# are replicated. Classes not listed (e.g. QKVParallelLinearWithLoRA, which interleaves q/k/v
# rows per rank) have no reviewed layout.
SHARDED_DIMS = {
    "ColumnParallelLinearWithLoRA": {"lora_b_stacked": -2},
    "MergedColumnParallelLinearWithLoRA": {"lora_b_stacked": -2},
    "MergedColumnParallelLinearVariableSliceWithLoRA": {"lora_b_stacked": -2},
    "MergedQKVParallelLinearWithLoRA": {"lora_b_stacked": -2},
    "RowParallelLinearWithLoRA": {"lora_a_stacked": -1},
    "ReplicatedLinearWithLoRA": {},
}


class VerifiedAdapterWorker(NativeSamplingWorker):
    def load_adapter(self, payload: list[Any]) -> bool:
        import msgspec
        from vllm.lora.request import LoRARequest

        # Untyped collective RPC decodes this array-like Struct as a plain list.
        request = msgspec.convert(payload, type=LoRARequest)
        return self.model_runner.add_lora(request)

    def verify_adapter(self, adapter_id: int, source: str, expected_sha: str) -> dict[str, Any]:
        tensors, scaling, digest = source_adapter(Path(source), expected_sha)
        manager = self.model_runner.lora_manager._adapter_manager
        adapter, slot = active_adapter(manager, adapter_id)
        payload = {
            "packed": manager.packed_modules,
            "modules": list(manager.modules),
            "cpu": cpu_buffers(adapter),
            "gpu": gpu_buffers(manager, slot),
        }
        report = verify_transfer(tensors, payload, scaling=scaling)
        if not report["cast_transfer_verified"]:
            raise ValueError(f"Active adapter transfer failed: {report['cast_failure_keys'][:5]}")
        report.update(adapter_id=adapter_id, slot=slot, native_sha256=digest)
        return report

    def adapter_shards(self, adapter_id: int) -> dict[str, Any]:
        """This rank's active buffers and which dimension tensor parallelism split."""
        require_pinned_vllm()
        manager = self.model_runner.lora_manager._adapter_manager
        if manager.lora_config.fully_sharded_loras:
            raise ValueError("Fully sharded LoRA buffers have an unreviewed layout")
        adapter, slot = active_adapter(manager, adapter_id)
        ranks = {module.tp_rank for module in manager.modules.values()}
        if len(ranks) != 1:
            raise ValueError(f"LoRA layers disagree on their tensor-parallel rank: {ranks}")
        rank = ranks.pop()
        return {
            "rank": rank,
            "adapter_id": adapter_id,
            "slot": slot,
            "packed": manager.packed_modules,
            "modules": list(manager.modules),
            "cpu": cpu_buffers(adapter) if rank == 0 else {},
            "gpu": gpu_buffers(manager, slot),
            "sharded_dims": {
                f"{name}.{label}": dim
                for name, module in manager.modules.items()
                for label, dim in sharded_dims(module).items()
            },
        }


def verify_tensor_parallel(
    shards: list[dict[str, Any]], source: Path, expected_sha: str, *, size: int
) -> dict[str, Any]:
    """Rebuild every rank's slices into full buffers and verify them against the adapter."""
    tensors, scaling, digest = source_adapter(source, expected_sha)
    report = verify_transfer(tensors, merge_tensor_parallel(shards, size=size), scaling=scaling)
    if not report["cast_transfer_verified"]:
        raise ValueError(f"Active adapter transfer failed: {report['cast_failure_keys'][:5]}")
    slots = {shard["slot"] for shard in shards}
    if len(slots) != 1 or {shard["adapter_id"] for shard in shards} != {shards[0]["adapter_id"]}:
        raise ValueError("Ranks hold the adapter in different slots")
    report.update(
        adapter_id=shards[0]["adapter_id"],
        slot=slots.pop(),
        native_sha256=digest,
        tensor_parallel_size=size,
    )
    return report


# --- Pinned vLLM layout


def source_adapter(path: Path, expected_sha: str) -> tuple[dict[str, Any], float, str]:
    require_pinned_vllm()
    from safetensors.torch import load_file

    digest = hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
    if digest != expected_sha:
        raise ValueError("Worker source adapter identity differs")
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("use_rslora") or config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("Unreviewed adapter scaling pattern")
    scaling = config["lora_alpha"] / config["r"]
    return load_file(str(path / "adapter_model.safetensors")), scaling, digest


def active_adapter(manager: Any, adapter_id: int) -> tuple[Any, int]:
    adapter = manager.get_adapter(adapter_id)
    if adapter is None or adapter.id != adapter_id or adapter_id not in manager.lora_index_to_id:
        raise ValueError("Requested adapter is not loaded and active")
    return adapter, manager.lora_index_to_id.index(adapter_id)


def cpu_buffers(adapter: Any) -> dict[str, list[Any]]:
    buffers = {}
    for name, weights in adapter.loras.items():
        for label in ("lora_a", "lora_b"):
            value = getattr(weights, label)
            values = value if isinstance(value, (tuple, list)) else [value]
            buffers[name + "." + label] = [
                value.detach().cpu() if value is not None else None for value in values
            ]
    return buffers


def gpu_buffers(manager: Any, slot: int) -> dict[str, list[Any]]:
    buffers = {}
    for name, module in manager.modules.items():
        for label in ("lora_a_stacked", "lora_b_stacked"):
            if hasattr(module, label):
                value = getattr(module, label)
                values = value if isinstance(value, (tuple, list)) else [value]
                buffers[name + "." + label] = [value[slot].detach().cpu() for value in values]
    return buffers


def sharded_dims(module: Any) -> dict[str, int]:
    name = type(module).__name__
    if name not in SHARDED_DIMS:
        raise ValueError(f"Unreviewed tensor-parallel LoRA layout: {name}")
    if name == "ColumnParallelLinearWithLoRA" and module.is_merged_col_linear:
        raise ValueError("One LoRA over gate/up interleaves rows per rank")
    if name == "MergedQKVParallelLinearWithLoRA" and module.base_layer.num_kv_head_replicas != 1:
        raise ValueError("Replicated KV heads give overlapping LoRA B slices")
    return SHARDED_DIMS[name]
