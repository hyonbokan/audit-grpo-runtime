"""Memory, time and hardware-tier estimates for LoRA GRPO with long reasoning traces.

Calibrated on measured fits (Qwen3.8-27B, 2 x RTX A6000, 2026-10-22). Estimates, not
measurements: a fit test on the target hardware decides. Weights are BF16 (no quantization).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GIB = 2**30

# --- Calibration (docs/compute-planning.md, "Measured on the A6000 host")

# Update sequence memory per GPU, GiB per 1k tokens, at hidden size 5120; scaled by hidden size.
SLOPE_GIB_PER_1K = {"untiled": 0.52, "tiled": 0.38}
CALIBRATION_HIDDEN = 5120
FIXED_GIB = 1.5  # CUDA context, allocator, LoRA r16 and optimizer state
OFFLOAD_STAGING_GIB = 1.8  # pinned-copy staging when layers stream from host RAM
SLOT_LAYERS = 8  # block-swap prefetch slots resident while streaming (measured layout)
# Rollout: measured 47-68 tok/s for 4 draws vs a ~114 tok/s bandwidth bound -> 0.5.
ROLLOUT_EFFICIENCY = 0.5
# Assumed decode latency floor per token per sequence (kernel launches, tensor-parallel sync).
DECODE_FLOOR_S = 0.006
ROLLOUT_COMPUTE_MFU = 0.3
UPDATE_MFU = {"A6000": 0.17}  # measured: layer split, all layers streamed, tiled MLP
SHARDED_UPDATE_MFU = 0.25  # assumed for H100-class GPUs with a sharded trainer
KV_HEADROOM = 0.9  # share of GPU memory a serving engine may use
DRAWS_PER_GROUP = 4
CONTRAST_SHARE = 0.5  # share of groups with a reward contrast (base draws: 42-46%)
PRACTICAL_FACTOR = 3  # judge, weight sync, reloads and framework overhead over the ideal time
MAX_GPUS = 64

TIERS = (
    {"name": "pilot", "groups_batched": 4, "offload": True, "updates": 100, "groups_per_update": 4},
    {
        "name": "standard",
        "groups_batched": 16,
        "offload": False,
        "updates": 100,
        "groups_per_update": 8,
    },
    {
        "name": "extended",
        "groups_batched": 32,
        "offload": False,
        "updates": 200,
        "groups_per_update": 16,
    },
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="candidate models: size, BF16 weights, KV cache")
    fit = sub.add_parser("fit", help="update memory per GPU for one configuration")
    fit.add_argument("--model", required=True)
    fit.add_argument("--gpu", required=True)
    fit.add_argument("--gpus", type=int, required=True)
    fit.add_argument("--tokens", type=int, required=True)
    fit.add_argument("--tiled", action="store_true")
    fit.add_argument("--offload", action="store_true")
    fit.add_argument("--cp", type=int, default=1, help="context-parallel degree")
    tiers = sub.add_parser("tiers", help="smallest GPU count per tier, time and GPU-hours")
    tiers.add_argument("--models", nargs="+", required=True)
    tiers.add_argument("--gpu-types", nargs="+", default=["H100", "H200", "B200", "B300"])
    tiers.add_argument("--prompt", type=int, default=30_000)
    tiers.add_argument("--output", type=int, default=60_000)
    args = parser.parse_args(argv)
    commands = {"models": print_models, "fit": print_fit, "tiers": print_tiers}
    print(commands[args.command](args, load_models(), load_gpus()))
    return 0


# --- Registries


def load_models() -> dict[str, dict[str, Any]]:
    return json.loads((HERE / "models.json").read_text())["models"]


def load_gpus() -> dict[str, dict[str, Any]]:
    return json.loads((HERE / "hardware.json").read_text())["gpus"]


def pick(registry: dict[str, Any], name: str) -> Any:
    if name not in registry:
        raise KeyError(f"Unknown {name!r}; known: {sorted(registry)}")
    return registry[name]


# --- Commands


def print_models(_args: argparse.Namespace, models: dict, _gpus: dict) -> str:
    rows = [
        "| Model | Total / active | BF16 weights | KV per 128k sequence | AA index | Licence |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        rows.append(
            f"| {name} | {m['total_b']:g}B / {m['active_b']:g}B | {weights_gib(m):,.0f} GiB | "
            f"{kv_gib(m, 128_000):.1f} GiB | {m['aa_index'] or '—'} | {m['licence']} |"
        )
    return "\n".join(rows)


def print_fit(args: argparse.Namespace, models: dict, gpus: dict) -> str:
    model, gpu = pick(models, args.model), pick(gpus, args.gpu)
    need = update_gib(model, gpu, args.gpus, args.tokens, args.tiled, args.offload, args.cp)
    verdict = "fits" if need <= gpu["gib"] else "does not fit"
    return f"{args.model} on {args.gpus} x {args.gpu}: {need:.1f} of {gpu['gib']} GiB ({verdict})"


def print_tiers(args: argparse.Namespace, models: dict, gpus: dict) -> str:
    rows = [
        f"Sequence: {args.prompt // 1000}k prompt + {args.output // 1000}k output; BF16 weights.\n",
        "| Model | GPU | Tier | GPUs | Update setup | Wall hours | GPU-hours | Host RAM / node |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in args.models:
        model = pick(models, name)
        for gpu_name in args.gpu_types:
            gpu = pick(gpus, gpu_name)
            for spec in TIERS:
                plan = tier(model, gpu_name, spec, args.prompt, args.output)
                if plan["gpus"] is None:
                    rows.append(f"| {name} | {gpu_name} | {spec['name']} | > {MAX_GPUS} | | | | |")
                    continue
                rows.append(
                    f"| {name} | {gpu_name} | {spec['name']} | {plan['gpus']} | {plan['setup']} | "
                    f"{plan['wall_hours']:,.0f} | {plan['gpu_hours']:,.0f} | "
                    f"{plan['host_tb_per_node']:.1f} of {gpu['host_ram_tb']} TB |"
                )
    return "\n".join(rows)


# --- Memory


def weights_gib(model: dict) -> float:
    return model["total_b"] * 1e9 * 2 / GIB


def kv_gib(model: dict, tokens: int) -> float:
    return model["kv_kib_per_token"] * 1024 * tokens / GIB


def update_gib(
    model: dict, gpu: dict, gpus: int, tokens: int, tiled: bool, offload: bool, cp: int = 1
) -> float:
    """Peak update memory per GPU: resident weight share + sequence memory / CP + fixed."""
    del gpu  # Memory per GPU does not depend on the GPU type; kept for a uniform signature.
    weights = weights_gib(model)
    if offload:
        resident = weights / model["layers"] * SLOT_LAYERS / gpus + OFFLOAD_STAGING_GIB
    else:
        resident = weights / gpus
    return resident + sequence_gib(model, tokens, tiled) / cp + FIXED_GIB


def sequence_gib(model: dict, tokens: int, tiled: bool) -> float:
    dense = SLOPE_GIB_PER_1K["tiled" if tiled else "untiled"] * tokens / 1000
    dense *= model["hidden"] / CALIBRATION_HIDDEN
    # MoE: routed hidden copies (in/out) and expert intermediates; unmeasured, an upper estimate.
    moe = model["topk"] * (4 * model["hidden"] + 6 * model["moe_intermediate"]) * tokens / GIB
    return dense + moe


def host_checkpoint_gib(model: dict, tokens: int) -> float:
    """Checkpointed layer inputs kept in host RAM for one sequence (bf16)."""
    return model["layers"] * model["hidden"] * 2 * tokens / GIB


def rollout_fits(model: dict, gpu: dict, gpus: int, sequences: int, tokens: int) -> bool:
    cache = sequences * kv_gib(model, tokens)
    return weights_gib(model) + cache <= KV_HEADROOM * gpu["gib"] * gpus


# --- Time


def rollout_seconds(model: dict, gpu: dict, gpus: int, sequences: int, tokens: int) -> float:
    """Decode time for `sequences` concurrent sequences of `tokens` output each."""
    experts = model.get("experts") or 0
    share = model["topk"] / experts if experts else 1.0
    routed = (model["total_b"] - model["active_b"]) / (1 - share) if experts else 0.0
    dense = model["active_b"] - routed * share
    touched = 1 - (1 - share) ** sequences if experts else 0.0
    step_bytes = 2e9 * (dense + routed * touched)
    step = max(
        DECODE_FLOOR_S,
        step_bytes / (ROLLOUT_EFFICIENCY * gpus * gpu["hbm_tbs"] * 1e12),
        sequences
        * 2e9
        * model["active_b"]
        / (ROLLOUT_COMPUTE_MFU * gpus * gpu["bf16_tflops"] * 1e12),
    )
    return tokens * step


def update_seconds(model: dict, gpu_name: str, gpu: dict, gpus: int, tokens: int) -> float:
    """One group's update: 4 draws x 6 x active params x tokens (attention not counted)."""
    mfu = UPDATE_MFU.get(gpu_name, SHARDED_UPDATE_MFU)
    computing = (
        1 if gpu_name == "A6000" else gpus
    )  # the A6000 layer split computes one GPU at a time
    flops = DRAWS_PER_GROUP * tokens * 6 * model["active_b"] * 1e9
    return flops / (mfu * computing * gpu["bf16_tflops"] * 1e12)


# --- Tiers


def tier(
    model: dict, gpu_name: str, spec: dict, prompt: int = 30_000, output: int = 60_000
) -> dict:
    """Smallest whole-node GPU count that fits rollout and update for this tier, with its time."""
    gpu = pick(load_gpus(), gpu_name)
    tokens = prompt + output
    sequences = DRAWS_PER_GROUP * spec["groups_batched"]
    for gpus in range(gpu["node_gpus"], MAX_GPUS + 1, gpu["node_gpus"]):
        if not rollout_fits(model, gpu, gpus, sequences, tokens):
            continue
        setup = update_setup(model, gpu, gpus, tokens, spec["offload"])
        if setup is None:
            continue
        offload, cp = setup
        per_group = rollout_seconds(model, gpu, gpus, sequences, output) / spec[
            "groups_batched"
        ] + update_seconds(model, gpu_name, gpu, gpus, tokens)
        groups = spec["updates"] * spec["groups_per_update"] / CONTRAST_SHARE
        wall = groups * per_group * PRACTICAL_FACTOR / 3600
        nodes = math.ceil(gpus / gpu["node_gpus"])
        pinned = weights_gib(model) / nodes if offload else 0.0
        checkpoints = gpu["node_gpus"] * host_checkpoint_gib(model, tokens) / cp
        return {
            "gpus": gpus,
            "setup": ("streamed" if offload else "resident") + (f", CP {cp}" if cp > 1 else ""),
            "wall_hours": wall,
            "gpu_hours": wall * gpus,
            "host_tb_per_node": (pinned + checkpoints) * GIB / 1e12,
        }
    return {"gpus": None}


def update_setup(model: dict, gpu: dict, gpus: int, tokens: int, allow_offload: bool):
    """(offload, CP degree) for the first configuration that fits, preferring resident weights."""
    cp = 1
    while cp <= gpus:
        for offload in (False, True) if allow_offload else (False,):
            if (
                update_gib(model, gpu, gpus, tokens, tiled=True, offload=offload, cp=cp)
                <= gpu["gib"]
            ):
                return offload, cp
        cp *= 2
    return None


if __name__ == "__main__":
    raise SystemExit(main())
