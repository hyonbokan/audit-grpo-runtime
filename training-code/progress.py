"""Training order and per-group update decisions."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from runtime_io import read_json, save_json, sha256_file

GROUP_SIZE = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("group-decision",))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    rewards = read_json(args.out / "grading/rewards.json")
    decision = group_decision(rewards)
    save_json(
        args.out / "group-decision.json",
        {
            "decision": decision,
            "rewards": rewards,
            "row_id": config["row_id"],
            "initial_step": config["initial_step"],
        },
    )
    print(decision)
    return 0


def group_decision(rewards: list[float]) -> str:
    if len(rewards) != GROUP_SIZE or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in rewards
    ):
        raise ValueError("Four finite known grades required")
    return "update" if len(set(rewards)) > 1 else "skip_no_reward_contrast"


def training_order(rows: list[str], *, epochs: int, seed: int) -> list[str]:
    if not rows or len(set(rows)) != len(rows) or not 1 <= epochs <= 2:
        raise ValueError("One or two finite shuffled passes of unique rows required")
    rng = random.Random(seed)
    result = []
    for _ in range(epochs):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        result.extend(shuffled)
    return result


def verify_dataset(config: dict[str, Any]) -> None:
    root = Path(config["dataset_host"])
    if sha256_file(root / "manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Frozen corpus manifest changed")
    for relative, digest in read_json(root / "manifest.json")["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"Frozen corpus file changed: {relative}")


if __name__ == "__main__":
    raise SystemExit(main())
