"""Pinned worker-buffer inspection; private LoRA layout access stays here."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from sampler_compat import SUPPORTED_VLLM_VERSION, NativeSamplingWorker
from transfer import verify_transfer


class VerifiedAdapterWorker(NativeSamplingWorker):
    def load_adapter(self, payload: list[Any]) -> bool:
        import msgspec
        from vllm.lora.request import LoRARequest

        # Untyped collective RPC decodes this array-like Struct as a plain list.
        request = msgspec.convert(payload, type=LoRARequest)
        return self.model_runner.add_lora(request)

    def verify_adapter(self, adapter_id: int, source: str, expected_sha: str) -> dict[str, Any]:
        if importlib.metadata.version("vllm") != SUPPORTED_VLLM_VERSION:
            raise ValueError("Adapter inspection requires pinned vLLM layout")
        from safetensors.torch import load_file

        path = Path(source)
        digest = hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
        if digest != expected_sha:
            raise ValueError("Worker source adapter identity differs")
        config = json.loads((path / "adapter_config.json").read_text())
        if config.get("use_rslora") or config.get("rank_pattern") or config.get("alpha_pattern"):
            raise ValueError("Unreviewed adapter scaling pattern")
        scaling = config["lora_alpha"] / config["r"]
        manager = self.model_runner.lora_manager._adapter_manager
        adapter = manager.get_adapter(adapter_id)
        if (
            adapter is None
            or adapter.id != adapter_id
            or adapter_id not in manager.lora_index_to_id
        ):
            raise ValueError("Requested adapter is not loaded and active")
        slot = manager.lora_index_to_id.index(adapter_id)
        payload = {
            "packed": manager.packed_modules,
            "modules": list(manager.modules),
            "cpu": {},
            "gpu": {},
        }
        for name, weights in adapter.loras.items():
            for label in ("lora_a", "lora_b"):
                value = getattr(weights, label)
                values = value if isinstance(value, (tuple, list)) else [value]
                payload["cpu"][name + "." + label] = [
                    value.detach().cpu() if value is not None else None for value in values
                ]
        for name, module in manager.modules.items():
            for label in ("lora_a_stacked", "lora_b_stacked"):
                if hasattr(module, label):
                    value = getattr(module, label)
                    values = value if isinstance(value, (tuple, list)) else [value]
                    payload["gpu"][name + "." + label] = [
                        value[slot].detach().cpu() for value in values
                    ]
        report = verify_transfer(
            load_file(str(path / "adapter_model.safetensors")), payload, scaling=scaling
        )
        if not report["cast_transfer_verified"]:
            raise ValueError(f"Active adapter transfer failed: {report['cast_failure_keys'][:5]}")
        report.update(adapter_id=adapter_id, slot=slot, native_sha256=digest)
        return report
