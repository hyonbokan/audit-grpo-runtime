"""Preserve native adapter values while naming the vLLM language-model path."""

import hashlib
import json
import shutil

import torch
from safetensors.torch import load_file, save_file


def prepare_adapter(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    original = load_file(str(source / "adapter_model.safetensors"))
    prefix = "base_model.model.model.layers."
    if not original or any(not key.startswith(prefix) for key in original):
        raise ValueError("Unreviewed native adapter prefix")
    mapped = {
        key.replace(prefix, "base_model.model.model.language_model.layers.", 1): value
        for key, value in original.items()
    }
    destination.mkdir()
    shutil.copyfile(source / "adapter_config.json", destination / "adapter_config.json")
    save_file(mapped, str(destination / "adapter_model.safetensors"))
    saved = load_file(str(destination / "adapter_model.safetensors"))
    if set(saved) != set(mapped) or not all(
        torch.equal(value, saved[key]) for key, value in mapped.items()
    ):
        raise ValueError("Adapter values changed during naming conversion")
    receipt = {
        "tensor_count": len(original),
        "all_tensor_values_identical": True,
        "native_sha256": hashlib.sha256(
            (source / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "mapped_sha256": hashlib.sha256(
            (destination / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "source": str(source),
        "destination": str(destination),
    }
    (destination / "mapping.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
