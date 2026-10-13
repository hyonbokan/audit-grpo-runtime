"""Seeded, reproducible project schedule for an unattended multi-group run."""

from __future__ import annotations

import random
from typing import Any


def project_of(row_id: str) -> str:
    return row_id.split("/", 1)[0]


def select_groups(
    *,
    train_rows: list[str],
    non_train_rows: list[str],
    requests: dict[str, dict[str, Any]],
    exclude_projects: set[str],
    seed: int,
    count: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    """Pick one random admitted group from each of `count` random unused projects.

    Projects that also own held-out rows, already-trained projects and prompts above the
    token cap are never eligible. No evaluation scores take part in the choice.
    """
    for row in train_rows:
        if row not in requests:
            raise ValueError(f"Row {row} has no verified request")
    blocked = set(exclude_projects) | {project_of(row) for row in non_train_rows}
    by_project: dict[str, list[str]] = {}
    for row in train_rows:
        if project_of(row) not in blocked and requests[row]["prompt_tokens"] <= max_prompt_tokens:
            by_project.setdefault(project_of(row), []).append(row)
    if count > len(by_project):
        raise ValueError(f"Requested {count} but only {len(by_project)} eligible projects")
    rng = random.Random(seed)
    projects = sorted(by_project)
    rng.shuffle(projects)
    order = [rng.choice(by_project[project]) for project in projects[:count]]
    return {
        "seed": seed,
        "selection_method": (
            "uniform shuffle of eligible unused projects, then uniform admitted group; "
            "excludes trained and held-out projects; no evaluation scores"
        ),
        "eligible_projects": len(by_project),
        "max_prompt_tokens": max_prompt_tokens,
        "order": order,
        "prompt_tokens": {row: requests[row]["prompt_tokens"] for row in order},
    }
