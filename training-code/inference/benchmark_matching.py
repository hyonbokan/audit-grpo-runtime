"""Delivered batch/vote/one-to-one matching, with transport failures kept unknown."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

Judge = Callable[[str], Awaitable[dict[str, Any]]]
BATCH_SIZE = 30


async def judge_vote(prompt: str, call: Judge) -> dict[str, Any]:
    async def attempt() -> dict[str, Any] | None:
        try:
            return await call(prompt)
        except Exception:
            # Transport owns failure/usage recording. Failed iterations do not cast votes.
            return None

    first = await asyncio.gather(attempt(), attempt())
    responses = [row for row in first if row is not None]
    if len(responses) == 2 and category(responses[0]) == category(responses[1]):
        return choose_vote(responses)
    third = await attempt()
    return choose_vote(responses + ([third] if third is not None else []))


def choose_vote(responses: list[dict[str, Any]]) -> dict[str, Any]:
    if not responses:
        raise ValueError("All judge iterations failed; grade unknown")
    counts = Counter(category(row) for row in responses)
    chosen = max((0, 1, 2), key=lambda kind: (counts[kind], kind))
    return next(row for row in responses if category(row) == chosen)


def category(row: dict[str, Any]) -> int:
    return 2 if row["is_match"] else 1 if row["is_partial_match"] else 0


def working_index(local: float, batch_length: int, batch_number: int) -> int:
    if type(local) not in (int, float) or int(local) != local or not 0 <= local < batch_length:
        return -1
    return int(local) + batch_number * BATCH_SIZE


async def evaluate_findings(
    findings: list[dict[str, Any]], truth: list[dict[str, Any]], template: str, judge: Judge
) -> dict[str, Any]:
    working = [dict(row, original_index=index) for index, row in enumerate(findings)]
    verdicts = []
    errors = []
    for target_index, target in enumerate(truth):
        try:
            result = await match_batches(working, target, template, judge)
        except Exception as error:
            errors.append(
                {"target_index": target_index, "error": f"{type(error).__name__}: {error}"}
            )
            # Later exact removals depend on this verdict; do not score a changed sequence.
            break
        index = result["index_of_finding_from_junior_auditor"]
        result["severity_from_truth"] = target["severity"]
        result["target_index"] = target_index
        result["index_of_finding_from_junior_auditor"] = (
            working[index]["original_index"] if index >= 0 else -1
        )
        if result["is_match"] and index >= 0:
            working.pop(index)
        verdicts.append(result)
    exact_indices = {
        row["index_of_finding_from_junior_auditor"]
        for row in verdicts
        if row["is_match"] and row["index_of_finding_from_junior_auditor"] >= 0
    }
    partial_indices = set()
    for row in verdicts:
        index = row["index_of_finding_from_junior_auditor"]
        if row["is_partial_match"] and not row["is_match"] and index >= 0:
            if index in exact_indices | partial_indices:
                row["is_partial_match"] = False
                row["explanation"] += " (Already consumed by an exact or earlier partial match.)"
            else:
                partial_indices.add(index)
    # A positive verdict with an unusable index cannot establish coverage.
    exact = sum(
        row["is_match"] and row["index_of_finding_from_junior_auditor"] >= 0 for row in verdicts
    )
    partial = sum(
        row["is_partial_match"]
        and not row["is_match"]
        and row["index_of_finding_from_junior_auditor"] >= 0
        for row in verdicts
    )
    return {
        "status": "unknown" if errors else "graded",
        "known_gt_count": len(truth),
        "exact_matches": exact,
        "partial_matches": partial,
        "exact_coverage": None if errors else exact / len(truth),
        "coverage_including_partial": None if errors else (exact + partial) / len(truth),
        "findings_count": len(findings),
        "native_metrics": None
        if errors
        else native_metrics(len(truth), len(findings), exact, partial),
        "unmatched_findings": None
        if errors
        else len(findings) - len(exact_indices | partial_indices),
        "verdicts": verdicts,
        "errors": errors,
    }


def native_metrics(known: int, scanned: int, exact: int, partial: int) -> dict[str, float]:
    precision = exact / scanned if scanned else 0.0
    recall = exact / known if known else 0.0
    with_partial = (exact + partial) / scanned if scanned else 0.0
    recall_with_partial = (exact + partial) / known if known else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision_with_partial": with_partial,
        "recall_with_partial": recall_with_partial,
        "f1_with_partial": 2
        * with_partial
        * recall_with_partial
        / (with_partial + recall_with_partial)
        if with_partial + recall_with_partial
        else 0.0,
    }


async def match_batches(
    working: list[dict[str, Any]], target: dict[str, Any], template: str, judge: Judge
) -> dict[str, Any]:
    best = None
    for start in range(0, len(working), BATCH_SIZE):
        batch = [
            {key: row[key] for key in ("issue", "description", "contracts", "severity")}
            for row in working[start : start + BATCH_SIZE]
        ]
        batch = [dict(row, index=index) for index, row in enumerate(batch)]
        prompt = template.replace(
            "{ truth_finding }", json.dumps(target, indent=2, ensure_ascii=False), 1
        )
        prompt = prompt.replace(
            "{ junior_findings }", json.dumps(batch, indent=2, ensure_ascii=False), 1
        )
        result = dict(await judge(prompt))
        result["index_of_finding_from_junior_auditor"] = working_index(
            result["index_of_finding_from_junior_auditor"], len(batch), start // BATCH_SIZE
        )
        if (result["is_match"] or result["is_partial_match"]) and result[
            "index_of_finding_from_junior_auditor"
        ] < 0:
            raise ValueError("Positive judge verdict has no valid finding index; grade unknown")
        if result["is_match"]:
            return result
        if best is None or (result["is_partial_match"] and not best["is_partial_match"]):
            best = result
    return (
        best
        if best is not None
        else {
            "is_match": False,
            "is_partial_match": False,
            "explanation": "No unconsumed findings",
            "severity_from_junior_auditor": "N/A",
            "severity_from_truth": target["severity"],
            "index_of_finding_from_junior_auditor": -1,
        }
    )
