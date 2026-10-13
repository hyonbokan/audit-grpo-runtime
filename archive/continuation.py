"""Frozen row schedule and checkpoint chaining for the bounded continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cycle_contract import validate_config
from inference.benchmark import validate_request


def validate_schedule(config: dict[str, Any]) -> None:
    validate_config(config)
    schedule = config["schedule"]
    if not isinstance(schedule, list) or not 1 <= len(schedule) <= 6:
        raise ValueError("One to six explicit training rows required")
    dataset = Path(config["dataset_workspace"])
    rows = [
        json.loads(line) for line in (dataset / "data/private/train.jsonl").read_text().splitlines()
    ]
    private = {row["row_id"]: row for row in rows}
    if len(private) != len(rows):
        raise ValueError("Duplicate private training row")
    for index, selected in enumerate(schedule):
        row = private.get(selected["row_id"])
        if row is None or not any(
            not target["required_context"] for target in row["scorer_metadata"]["targets"]
        ):
            raise ValueError("Scheduled row has no eligible code-only private targets")
        path = Path(selected["request"])
        request = json.loads(path.read_text())
        if request["row_id"] != selected["row_id"]:
            raise ValueError("Request row differs from scheduled row")
        validate_request(request)
        if (
            request["max_model_len"] != selected["context"]
            or request["full_completion_ceiling"] != config["sampling"]["max_tokens"]
            or request["model"]["id"] != config["model_id"]
            or request["model"]["revision"] != config["model_revision"]
        ):
            raise ValueError("Scheduled request contract differs")
        if (
            selected.get("request_sha256")
            and hashlib.sha256(path.read_bytes()).hexdigest() != (selected["request_sha256"])
        ):
            raise ValueError("Frozen request identity differs")
        initial = config["initial_step"] + index
        validate_config(
            {
                **config,
                "initial_step": initial,
                "target_step": initial + 1,
                "engine": {**config["engine"], "max_model_len": selected["context"]},
            }
        )


def prepare_step(config: dict[str, Any], out: Path, index: int) -> Path:
    schedule = config["schedule"]
    if type(index) is not int or not 0 <= index < len(schedule) <= 6:
        raise ValueError("Step is outside the frozen schedule")
    initial = config["initial_step"] + index
    checkpoint = Path(config["checkpoint"])
    if index:
        previous = out / f"steps/step-{initial}"
        result = json.loads((previous / "cycle-result.json").read_text())
        if not result["decision"]["approved"] or result["target_step"] != initial:
            raise ValueError("Predecessor update was not accepted")
        checkpoint = previous / f"update/checkpoint-{initial}"
        if json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] != initial:
            raise ValueError("Predecessor checkpoint progression differs")
        digest = hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
        receipt = json.loads((previous / "update/receipt.json").read_text())
        if (
            receipt["adapter_sha256"] != digest
            or receipt["weight_version"] != f"step-{initial}:{digest}"
        ):
            raise ValueError("Accepted predecessor checkpoint identity differs")
    selected = schedule[index]
    step = {key: value for key, value in config.items() if key != "schedule"}
    step.update(
        initial_step=initial,
        target_step=initial + 1,
        row_id=selected["row_id"],
        request=selected["request"],
        checkpoint=str(checkpoint),
        initial_adapter_sha256=(
            hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest()
            if index
            else config["initial_adapter_sha256"]
        ),
        engine={**config["engine"], "max_model_len": selected["context"]},
    )
    validate_config(step)
    destination = out / f"steps/step-{initial + 1}"
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "config.json"
    path.write_text(json.dumps(step, indent=2) + "\n")
    return path


def record_progress(config: dict[str, Any], out: Path) -> dict[str, Any]:
    steps = []
    for index, selected in enumerate(config["schedule"]):
        target = config["initial_step"] + index + 1
        root = out / f"steps/step-{target}"
        if not (root / "cycle-result.json").exists():
            break
        result = json.loads((root / "cycle-result.json").read_text())
        if not result["decision"]["approved"]:
            break
        rollout = json.loads((root / "rollout/receipt.json").read_text())
        update = json.loads((root / "update/receipt.json").read_text())
        measured = json.loads((root / "grading/metrics.json").read_text())
        if measured["row_id"] != selected["row_id"]:
            raise ValueError("Recorded grading row differs from schedule")
        seconds_path = root / "step-seconds.txt"
        charges = []
        for receipt_path in sorted((root / "grading").glob("call-*/budget-reservation.json")):
            receipt = json.loads(receipt_path.read_text())
            ledger = json.loads(Path(receipt["ledger"]).read_text())
            charges.append(
                next(call for call in ledger["calls"] if call["id"] == receipt["reservation_id"])
            )
        steps.append(
            {
                "row_id": selected["row_id"],
                "optimizer_step": target,
                "checkpoint": str(root / f"update/checkpoint-{target}"),
                "rewards": json.loads((root / "grading/rewards.json").read_text()),
                "training_metrics": measured,
                "completion_tokens": rollout["completion_tokens"],
                "generation_seconds": rollout["generation_seconds"],
                "runtime_seconds": int(seconds_path.read_text()) if seconds_path.exists() else None,
                "judge_budget_charge_usd": sum(call["budget_charge_usd"] for call in charges),
                "judge_known_usage_estimate_usd": sum(
                    call["estimated_usd"] or 0 for call in charges
                ),
                "judge_unknown_usage_calls": sum(call["usage"] is None for call in charges),
                **{
                    key: update[key]
                    for key in (
                        "loss",
                        "gradient_norm",
                        "correction_clip_fraction",
                        "correction_ess_fraction",
                        "weight_version",
                        "adapter_sha256",
                    )
                },
            }
        )
    sampled = [step["row_id"] for step in steps]
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "completed_updates": len(steps),
        "sampled_row_ids": sampled,
        "unsampled_row_ids": [
            row["row_id"] for row in config["schedule"] if row["row_id"] not in sampled
        ],
        "last_checkpoint": steps[-1]["checkpoint"] if steps else config["checkpoint"],
        "completion_tokens": sum(step["completion_tokens"] for step in steps),
        "steps": steps,
        "judge_budget_charge_usd": sum(step["judge_budget_charge_usd"] for step in steps),
        "judge_known_usage_estimate_usd": sum(
            step["judge_known_usage_estimate_usd"] for step in steps
        ),
    }
    temporary = out / "progress.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "progress.json")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--seconds-left", type=int)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.record:
        print(json.dumps(record_progress(config, args.out)))
        return 0
    if args.seconds_left is not None:
        config["seconds_limit"] = min(config["seconds_limit"], args.seconds_left)
    print(prepare_step(config, args.out, args.index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
