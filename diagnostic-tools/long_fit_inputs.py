"""Seed a long-sequence update fit: recorded draws cut to a target length, synthetic grades.

`build` writes the cycle config into the run directory and stages rollout and grading beside it
(`cycle_contract prepare` needs a fresh run directory); `seed` moves them in after prepare.
No judge call: the staged grades are rewards [1, 0, 0, 0]. Recorded draws without sampling
support get a synthetic two-token support and zero sampled log-probs, so correction ratios
stay <= 1; such a fit measures memory and time, not correction health.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

REWARDS = [1.0, 0.0, 0.0, 0.0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_args = sub.add_parser("build", help="write config.json and stage rollout + grading")
    build_args.add_argument("--template", type=Path, required=True, help="a visit's config.json")
    build_args.add_argument("--samples", type=Path, required=True, help="recorded samples.json")
    build_args.add_argument("--run", type=Path, required=True, help="new run directory")
    build_args.add_argument("--length", type=int, required=True, help="prompt + completion")
    build_args.add_argument("--vocab", type=int, required=True, help="model vocabulary size")
    build_args.add_argument("--schedule-id", required=True)
    build_args.add_argument("--training-cutoff", required=True)
    build_args.add_argument("--seconds-limit", type=int, required=True)
    build_args.add_argument("--offload-layers", type=int, required=True)
    build_args.add_argument("--tiled-mlp", choices=("true", "false"), required=True)
    build_args.add_argument("--update-placement", required=True, help="JSON object")
    seed_args = sub.add_parser("seed", help="move the staged phases into the prepared run")
    seed_args.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    {"build": run_build, "seed": lambda parsed: seed(parsed.run)}[args.command](args)
    return 0


def run_build(args: argparse.Namespace) -> None:
    settings = {
        "schedule_id": args.schedule_id,
        "training_cutoff": args.training_cutoff,
        "seconds_limit": args.seconds_limit,
        "offload_layers": args.offload_layers,
        "tiled_mlp": args.tiled_mlp == "true",
        "update_placement": json.loads(args.update_placement),
    }
    template = json.loads(args.template.read_text())
    draws = json.loads(args.samples.read_text())
    built = build(template, settings, draws, args.length, args.vocab)
    args.run.mkdir(parents=True, exist_ok=False)
    (args.run / "config.json").write_text(json.dumps(built["config"], indent=2) + "\n")
    write_staged(built, args.run)
    print("sequence tokens:", built["receipt"]["sequence_tokens"])


def build(
    template: dict[str, Any],
    settings: dict[str, Any],
    draws: list[dict[str, Any]],
    length: int,
    vocab: int,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config.update(
        schedule_id=settings["schedule_id"],
        absolute_deadline=settings["training_cutoff"],
        seconds_limit=settings["seconds_limit"],
        offload_layers=settings["offload_layers"],
        tiled_mlp=settings["tiled_mlp"],
        update_placement=settings["update_placement"],
    )
    config["continuation_schedule"]["training_cutoff"] = settings["training_cutoff"]
    version = f"step-{config['initial_step']}:{config['initial_adapter_sha256']}"
    samples = [cut(draw, length, vocab, config["row_id"], version) for draw in draws]
    receipt = {
        "status": "versioned_rollout_saved",
        "weight_version": version,
        "synthetic_fit_input": True,
        "sequence_tokens": [len(s["prompt_ids"]) + len(s["completion_ids"]) for s in samples],
    }
    return {"config": config, "samples": samples, "receipt": receipt, "rewards": list(REWARDS)}


def cut(draw: dict[str, Any], length: int, vocab: int, row_id: str, version: str) -> dict:
    keep = min(len(draw["completion_ids"]), length - len(draw["prompt_ids"]))
    if keep < 1:
        raise ValueError(f"Length {length} is shorter than the prompt ({len(draw['prompt_ids'])})")
    ids = draw["completion_ids"][:keep]
    return {
        "row_id": row_id,
        "prompt_ids": draw["prompt_ids"],
        "completion_ids": ids,
        "sampling_support_ids": [[token, (token + 1) % vocab] for token in ids],
        "logprobs": [0.0] * keep,
        "logprob_semantics": "sampling_processed",
        "weight_version": version,
        "weight_representation": "runtime_lora",
        "finish_reason": draw["finish_reason"] if keep == len(draw["completion_ids"]) else "length",
        "raw_text": "",
        "reasoning": "",
        "final_text": "",
        "raw_logprobs": None,
    }


def staging(run: Path) -> Path:
    return run.with_name(run.name + ".fit-seed")


def write_staged(built: dict[str, Any], run: Path) -> None:
    stage = staging(run)
    (stage / "rollout").mkdir(parents=True, exist_ok=False)
    (stage / "grading").mkdir()
    (stage / "rollout/samples.json").write_text(json.dumps(built["samples"]))
    (stage / "rollout/receipt.json").write_text(json.dumps(built["receipt"], indent=2))
    (stage / "grading/rewards.json").write_text(json.dumps(built["rewards"]))


def seed(run: Path) -> None:
    stage = staging(run)
    for phase in ("rollout", "grading"):
        if (run / phase).exists():
            raise FileExistsError(f"{run / phase} already exists")
        shutil.move(str(stage / phase), str(run / phase))
    stage.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
