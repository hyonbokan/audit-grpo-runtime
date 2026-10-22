"""Reviewed multi-visit training schedules: shared defaults plus per-schedule overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REGISTRY = Path(__file__).with_name("schedules.json")


def scope(config: dict[str, Any]) -> dict[str, Any]:
    """The reviewed schedule a run or cycle config names; unnamed or unknown ids fail."""
    if config.get("schedule_id") is None:
        raise ValueError("Config names no schedule_id")
    return schedule(config["schedule_id"])


def schedule(schedule_id: str) -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text())
    if schedule_id not in registry["schedules"]:
        raise ValueError(
            f"Unreviewed schedule {schedule_id!r}; registered: {sorted(registry['schedules'])}"
        )
    resolved = merge(registry["defaults"], registry["schedules"][schedule_id])
    resolved["schedule_id"] = schedule_id
    resolved["training"]["schedule_id"] = schedule_id
    return resolved


def schedule_after_update(config: dict) -> dict:
    """The continuation schedule saved beside a checkpoint, advanced past this update."""
    schedule = dict(config["continuation_schedule"])
    schedule.update(
        initial_step=config["initial_step"],
        target_step=config["target_step"],
        learning_rate=config["learning_rate"],
        warmup_steps=0,
        updates_remaining=schedule["maximum_step"] - config["target_step"],
        cursor=schedule["cursor"] + 1,
    )
    return schedule


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(base[key], value) if isinstance(value, dict) and key in base else value
    return result
