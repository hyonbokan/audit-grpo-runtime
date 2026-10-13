"""Create an inference-only BF16 model without mutating resumable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_checkpoint(
    base: Path, adapter: Path, output: Path, revision: str
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    if output.exists():
        raise FileExistsError(output)
    config = json.loads((adapter / "adapter_config.json").read_text())
    unsupported = (
        "use_rslora",
        "use_dora",
        "use_qalora",
        "fan_in_fan_out",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "lora_bias",
        "target_parameters",
        "alora_invocation_tokens",
        "layer_replication",
    )
    if config["peft_type"] != "LORA" or config.get("bias", "none") != "none":
        raise ValueError("Unsupported adapter type or bias")
    if any(config.get(key) for key in unsupported):
        raise ValueError("Unsupported adapter feature/scaling pattern")
    rank = config["r"]
    if rank <= 0:
        raise ValueError("Adapter rank must be positive")
    scale = config["lora_alpha"] / rank
    adapter_weights = adapter / "adapter_model.safetensors"
    adapter_digest = sha256(adapter_weights)
    state = load_file(str(adapter_weights))
    prefix = "base_model.model.model.layers."
    targets = {}
    used = set()
    for key, value in state.items():
        if not key.startswith(prefix) or not key.endswith(".lora_A.weight"):
            continue
        partner = key.replace(".lora_A.weight", ".lora_B.weight")
        if (
            partner not in state
            or value.shape[0] != rank
            or state[partner].shape[1] != rank
        ):
            raise ValueError(f"Invalid adapter pair: {key}")
        target = key.replace(prefix, "model.language_model.layers.", 1)
        target = target.replace(".lora_A.weight", ".weight")
        targets[target] = (value, state[partner])
        used.update((key, partner))
    if not targets or used != set(state):
        raise ValueError("Unmapped or unpaired adapter tensors")
    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if not set(targets).issubset(index["weight_map"]):
        raise ValueError("Adapter target missing from base weight index")
    started = time.monotonic()
    output.mkdir(parents=True)
    manifest = {
        "status": "merging",
        "artifact_role": "inference_only_not_resumable_training",
        "base_path": str(base),
        "base_revision": revision,
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_digest,
        "adapter_config": config,
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "rank": rank,
        "scale": scale,
        "stored_weight_precision": "bfloat16",
        "adapter_operand_precision": "bfloat16",
        "accumulation_precision": "float32",
        "method": "BF16(FP32(W_BF16) + scale * FP32(B_BF16) @ FP32(A_BF16))",
        "row_block_size": 256,
        "merged_modules": 0,
        "files": {},
        "targets": [],
    }
    manifest_path = output / "MERGE_MANIFEST.json"
    try:
        for filename in sorted(set(index["weight_map"].values())):
            source = base / filename
            source_hash = sha256(source)
            with safe_open(str(source), framework="pt", device="cpu") as reader:
                tensors = {name: reader.get_tensor(name) for name in reader.keys()}
                metadata = reader.metadata()
            changed = []
            for name, weight in tensors.items():
                if name not in targets:
                    continue
                a, b = targets[name]
                if weight.dtype != torch.bfloat16 or weight.shape != (
                    b.shape[0],
                    a.shape[1],
                ):
                    raise ValueError(f"Base dtype/shape differs for {name}")
                a = a.to(torch.bfloat16).float()
                b = b.to(torch.bfloat16).float()
                merged = torch.empty_like(weight)
                changed_elements = 0
                for row in range(0, weight.shape[0], 256):
                    end = row + 256
                    block = weight[row:end].float() + (b[row:end] @ a) * scale
                    merged[row:end] = block.to(torch.bfloat16)
                    changed_elements += int((merged[row:end] != weight[row:end]).sum())
                tensors[name] = merged
                changed.append(name)
                manifest["targets"].append(
                    {
                        "name": name,
                        "shape": list(weight.shape),
                        "changed_elements": changed_elements,
                    }
                )
            destination = output / filename
            save_file(tensors, str(destination), metadata=metadata)
            del tensors
            # Re-read every output tensor: untouched tensors must retain exact values.
            with safe_open(str(source), framework="pt", device="cpu") as original:
                with safe_open(str(destination), framework="pt", device="cpu") as saved:
                    if set(original.keys()) != set(saved.keys()):
                        raise ValueError("Saved shard tensor keys changed")
                    for name in original.keys():
                        if name not in targets and not torch.equal(
                            original.get_tensor(name), saved.get_tensor(name)
                        ):
                            raise ValueError(f"Untouched tensor changed: {name}")
            if sha256(source) != source_hash:
                raise ValueError(f"Original base changed: {source}")
            manifest["files"][filename] = {
                "source_sha256": source_hash,
                "merged_sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "updated_tensors": changed,
            }
            manifest["merged_modules"] += len(changed)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "event": "shard_saved",
                        "file": filename,
                        "merged_modules": manifest["merged_modules"],
                    }
                ),
                flush=True,
            )
        for source in sorted(base.iterdir()):
            if not source.is_file() or source.name.endswith(".safetensors"):
                continue
            shutil.copyfile(source, output / source.name)
            manifest["files"][source.name] = {
                "source_sha256": sha256(source),
                "merged_sha256": sha256(output / source.name),
                "bytes": (output / source.name).stat().st_size,
            }
        if sha256(adapter_weights) != adapter_digest:
            raise ValueError("Original adapter changed")
        if manifest["merged_modules"] != len(targets):
            raise ValueError("Not all adapter target tensors were merged")
        manifest.update(
            status="merged_and_verified", seconds=time.monotonic() - started
        )
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    merge_checkpoint(args.base, args.adapter, args.output, args.revision)


if __name__ == "__main__":
    main()
