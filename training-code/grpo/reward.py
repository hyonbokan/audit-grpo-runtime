"""Free-text reward: recall of the known findings, with no gain from listing more."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import benchmark_free_text as free_text
import free_text_corpus
from benchmark_matching import Judge
from grpo.budget import BudgetExhausted
from grpo.samples import write_new
from runtime_io import read_json, sha256_bytes

REWARD_KEYS = {"version", "volume_allowance"}


async def grade_group(
    samples: list[dict[str, Any]],
    row: dict[str, Any],
    reward: dict[str, Any],
    template: str,
    extract: Judge,
    match: Judge,
    new_attempt: Callable[[], None],
    out: Path,
    *,
    num_generations: int,
    max_retries: int = 1,
) -> list[float]:
    """Grade every draw; an unknown grade regrades the whole group once, then withholds it."""
    if max_retries not in (0, 1):
        raise ValueError("A group is regraded at most once")
    if len(samples) != num_generations or num_generations < 2:
        raise ValueError("Incomplete rollout group")
    if {sample["row_id"] for sample in samples} != {row["row_id"]}:
        raise ValueError("Mixed row identities in rollout group")
    validate_reward(reward)
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "samples.json", samples)
    for attempt in range(1, max_retries + 2):
        folder = out / f"attempt-{attempt}"
        folder.mkdir()
        new_attempt()
        grades = await grade_draws(samples, row, reward, template, extract, match)
        write_new(folder / "grades.json", grades)
        if all(grade["reward"] is not None for grade in grades):
            return [grade["reward"] for grade in grades]
    raise GroupWithheld(f"Entire rollout group withheld after {max_retries + 1} attempts: {out}")


class GroupWithheld(RuntimeError):
    pass


async def grade_draws(samples, row, reward, template, extract, match) -> list[dict[str, Any]]:
    results = await asyncio.gather(
        *(grade_one(sample, row, reward, template, extract, match) for sample in samples),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BudgetExhausted):
            raise result
    grades = []
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            # Provider and extraction failures leave the grade unknown, never zero.
            result = {
                "status": "unknown",
                "reward": None,
                "error": f"{type(result).__name__}: {result}",
            }
        grades.append({**result, "sample_index": index})
    return grades


async def grade_one(sample, row, reward, template, extract, match) -> dict[str, Any]:
    grade = await free_text.grade_draw(sample, row["truth"], template, extract, match)
    allowance = reward["volume_allowance"]
    factor = volume_factor(grade.get("findings_count"), allowance)
    return {**grade, "volume_factor": factor, "reward": draw_reward(grade, factor)}


def draw_reward(grade: dict[str, Any], factor: float | None) -> float | None:
    """Exact recall times the volume factor; a truncated draw scores 0 and others stay unknown."""
    if grade["status"] == "truncated":
        return 0.0
    if grade["status"] != "graded":
        return None
    return grade["exact_matches"] / grade["known_gt_count"] * factor


def volume_factor(findings: int | None, allowance: int) -> float | None:
    if findings is None:
        return None
    return min(1.0, allowance / findings) if findings else 0.0


# --- Contract


def validate_reward(reward: dict[str, Any]) -> None:
    if set(reward) != REWARD_KEYS:
        raise ValueError(f"Unknown or missing reward settings: {sorted(set(reward) ^ REWARD_KEYS)}")
    if not isinstance(reward["version"], str) or not reward["version"]:
        raise ValueError("reward.version must name the contract version")
    allowance = reward["volume_allowance"]
    if allowance is None:
        raise ValueError("reward.volume_allowance is not set; decide O1 from the medium base run")
    if type(allowance) is not int or allowance < 1:
        raise ValueError("reward.volume_allowance must be an integer of at least 1")


def contract_sha256(reward: dict[str, Any], template: str) -> str:
    """Identity of everything that decides a reward: the formula, its setting and the judge."""
    scope = read_json(free_text.SCOPE_PATH)
    identity = {
        "reward": reward,
        "judge": scope["benchmark_judge"],
        "judge_prompt_sha256": sha256_bytes(template.encode()),
    }
    return sha256_bytes(json.dumps(identity, sort_keys=True).encode())


# --- Corpus


def load_split(corpus: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Policy records (prompt only) and the private rows, joined and checked."""
    if split not in free_text_corpus.SPLITS:
        raise ValueError(f"Unknown split {split!r}; known: {free_text_corpus.SPLITS}")
    private = {
        row["row_id"]: row
        for row in free_text_corpus.read_jsonl(corpus / f"data/private/{split}.jsonl")
    }
    policy = free_text_corpus.read_jsonl(corpus / f"data/policy/{split}.jsonl")
    if any(set(row) != {"row_id", "messages"} for row in policy):
        raise ValueError("Private fields in policy dataset")
    if {row["row_id"] for row in policy} != set(private) or len(policy) != len(private):
        raise ValueError("Private/policy row join differs")
    for row in policy:
        if row["messages"] != free_text_corpus.policy_messages(private[row["row_id"]]):
            raise ValueError("Saved policy prompt differs from code-only rendering")
    return [{"row_id": r["row_id"], "prompt": r["messages"]} for r in policy], private


Grader = Callable[..., Awaitable[list[float]]]
