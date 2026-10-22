"""Repeat the one-shot Markdown evaluation with fresh seeds in new result directories."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import schedules
from provenance import checksums
from runtime_io import read_json, save_json


def repeat_child_config(
    source: dict[str, Any], run_id: str, seed: int, schedule_id: str
) -> dict[str, Any]:
    reviewed = schedules.schedule(schedule_id)
    child = copy.deepcopy(source)
    child.pop("repeat_evaluation", None)
    child.update(
        run_id=run_id,
        weekend_schedule_id=schedule_id,
        absolute_deadline=reviewed["final_deadline"],
        seconds_limit=reviewed["evaluation_seconds_limit"],
        repeat_evaluation={"seed": seed, "arms": list(source["generation_order"])},
    )
    child["sampling"]["seed"] = seed
    child["engine"]["seed"] = seed
    return child


def write_repeat(
    runtime: Path, source_suite: str, prefix: str, seed: int, schedule_id: str
) -> list[str]:
    from benchmark_evaluation import validate_benchmark_config

    results = runtime / "results"
    sums = checksums()
    suite = read_json(results / source_suite / "config.json")
    reviewed = schedules.schedule(schedule_id)
    roots = []
    for project in suite["projects"]:
        run_id = f"{prefix}-evaluation-{project['name']}"
        source = read_json(results / project["run_id"] / "config.json")
        child = repeat_child_config(source, run_id, seed, schedule_id)
        validate_benchmark_config(child)
        root = results / run_id
        root.mkdir(parents=True, exist_ok=False)
        (root / "SOURCE_SHA256SUMS").write_text(sums)
        save_json(root / "config.json", child)
        project.update(run_id=run_id, run_root=f"/results/{run_id}")
        roots.append(str(root))
    suite.update(run_id=f"{prefix}-evaluation", absolute_deadline=reviewed["final_deadline"])
    root = results / f"{prefix}-evaluation"
    root.mkdir(parents=True, exist_ok=False)
    (root / "SOURCE_SHA256SUMS").write_text(sums)
    save_json(root / "config.json", suite)
    return [str(root), *roots]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source-suite", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--schedule", required=True)
    args = parser.parse_args()
    for path in write_repeat(
        args.runtime, args.source_suite, args.prefix, args.seed, args.schedule
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
