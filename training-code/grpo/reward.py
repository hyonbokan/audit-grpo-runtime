"""Private reward joins and bounded whole-group judging for the first pilot."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from core.schemas.finding_description import description_to_text
from core.schemas.llm_schema import LLMFindingList
from pydantic import ValidationError
from scripts.finetuning.tools.b_rollout_pilot import _grader
from scripts.finetuning.tools.calibrate_reward import scalar_reward
from scripts.finetuning.tools.reward_schema_candidate import MatchResponse, score_matching
from scripts.finetuning.tools.training_rows import render_training_row

from grpo.budget import BudgetExhausted
from grpo.samples import capture_sample as capture_sample
from grpo.samples import write_new


async def grade_group(samples, private, contract, judge, out, *, num_generations, max_retries=1):
    """Retry judging the same complete group; withhold it before any optimizer update."""
    if max_retries not in (0, 1):
        raise ValueError("Pilot permits at most one whole-group retry")
    if len(samples) != num_generations or num_generations < 2:
        raise ValueError("Incomplete rollout group")
    if len({sample["row_id"] for sample in samples}) != 1:
        raise ValueError("Mixed row identities in rollout group")
    if samples[0]["row_id"] not in private:
        raise ValueError("Unknown private row identity")
    validate_contract(contract)
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / "samples.json", samples)
    for attempt in range(1, max_retries + 2):
        folder = out / f"attempt-{attempt}"
        folder.mkdir()
        if hasattr(judge, "new_attempt"):
            judge.new_attempt()
        grades = []
        for index, sample in enumerate(samples):
            grade = await _grade(sample, private[sample["row_id"]], contract, judge)
            grade["sample_index"] = index
            grades.append(grade)
        write_new(folder / "grades.json", grades)
        if all(grade["reward"] is not None for grade in grades):
            return [grade["reward"] for grade in grades]
    raise GroupWithheld(f"Entire rollout group withheld after {max_retries + 1} attempts: {out}")


class GroupWithheld(RuntimeError):
    pass


def load_split(workspace: Path, split: str):
    if split not in ("train", "calibration", "evaluation"):
        raise ValueError("Unknown split")
    raw = [
        json.loads(line)
        for line in (workspace / f"data/private/{split}.jsonl").read_text().splitlines()
    ]
    policy = [
        json.loads(line)
        for line in (workspace / f"data/policy/{split}.jsonl").read_text().splitlines()
    ]
    rendering = None
    if (workspace / "manifest.json").is_file():
        rendering = json.loads((workspace / "manifest.json").read_text()).get(
            "policy_rendering_sha256"
        )
    private = {}
    for row in raw:
        if (
            row["row_id"] in private
            or row["profile"] != "none"
            or row["row_id"].rsplit("/", 1)[-1] != row["group_id"]
        ):
            raise ValueError("Invalid private row identity/profile/scope")
        rendered = render_training_row(row, include_context=False)
        if rendering:
            # Training-code root module; the judge container copies the whole tree.
            import audited_corpus

            frozen = audited_corpus.policy_rendering_bytes()
            if hashlib.sha256(frozen).hexdigest() != rendering:
                raise ValueError("Frozen policy rendering identity differs")
            rendered["messages"] = audited_corpus.policy_messages(row)
        private[row["row_id"]] = {**rendered, "source_files": row["source_files"]}
    if len(policy) != len(private) or {row["row_id"] for row in policy} != set(private):
        raise ValueError("Private/policy row join differs")
    records = []
    for row in policy:
        if set(row) != {"row_id", "messages"}:
            raise ValueError("Private fields in policy dataset")
        validate_policy(row["messages"], private[row["row_id"]])
        records.append({"row_id": row["row_id"], "prompt": row["messages"]})
    return records, private


def validate_policy(messages, private_row):
    if messages != private_row["messages"]:
        raise ValueError("Saved policy prompt differs from code-only rendering")


def validate_contract(contract):
    selected = contract["reward"]
    if {key: selected[key] for key in ("beta", "volume_cost", "max_findings", "format_floor")} != {
        "beta": 0.25,
        "volume_cost": 0.05,
        "max_findings": 32,
        "format_floor": -2,
    }:
        raise ValueError("Pilot scalar differs from selected contract")
    # Historical controls use high; the live user-selected contract may use medium.
    if contract["judge"].get("model") != "gpt-6.1-sol" or contract["judge"].get(
        "reasoning_effort"
    ) not in ("medium", "high"):
        raise ValueError("Pilot requires Sol with selected medium/high reasoning")


# --- Two-stage reward


async def _grade(sample, private_row, contract, judge):
    config = {
        key: contract["reward"][key]
        for key in ("beta", "volume_cost", "max_findings", "format_floor")
    }
    result = {
        "reward": None,
        "components": {
            "format_valid": True,
            "raw_findings": 0,
            "hits": None,
            "quality_sum": None,
            "withheld_reasons": [],
        },
        "judge_requests": [],
        "judge_responses": [],
    }
    components = result["components"]
    try:
        text = sample["final_text"].strip()
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[len("```json\n") : -3].strip()
        if sample["finish_reason"] != "stop":
            raise ValueError("Truncated completion")
        final = LLMFindingList.model_validate_json(text)
    except (ValidationError, ValueError):
        components["format_valid"] = False
        result["reward"] = scalar_reward(components, **config)
        return result
    components["raw_findings"] = len(final.findings)
    if not final.findings or len(final.findings) > config["max_findings"]:
        components.update(hits=0, quality_sum=0)
        result["reward"] = scalar_reward(components, **config)
        return result
    grader = _grader()
    candidates = [
        {
            **finding.model_dump(mode="json"),
            "id": f"c{index}",
            "description": description_to_text(finding.description),
        }
        for index, finding in enumerate(final.findings)
    ]
    case = {
        "case_id": sample["row_id"],
        "reasoning_kind": "sampled" if sample["reasoning"] else "unavailable",
        "payload": {
            "candidate_findings": candidates,
            "gt_findings": [
                {**copy.deepcopy(target["finding"]), "id": target["id"]}
                for target in private_row["scorer_metadata"]["eligible_targets"]
            ],
            "source_files": copy.deepcopy(private_row["source_files"]),
            "reasoning_trace": sample["reasoning"],
        },
    }
    result["case"] = case
    try:
        match_job = grader.match_job(case, 1)
        response = await _request(match_job, contract, judge, result)
        matching = score_matching(match_job["original_payload"], response)
        match = {
            "status": matching["status"],
            "condition": "base_1",
            "response": matching.get("normalized_response"),
            "matching": matching["matching"],
        }
        result["matching_errors"] = matching["errors"]
        reason_job = grader.reason_job(case, match)
        reason = None
        if reason_job:
            response = await _request(reason_job, contract, judge, result)
            parsed = grader.SCHEMAS["sgr_reason"].model_validate(response).model_dump()
            errors = grader.sgr_reward.validate_response(
                "sgr_reason", reason_job["payload"], parsed
            )
            reason = {
                "status": "invalid" if errors else "ok",
                "reasoning": grader.sgr_reward.reasoning_summary(parsed),
            }
            result["reasoning_errors"] = errors
        components.update(grader.completion_summary(case, match, reason))
    except BudgetExhausted:
        raise
    except Exception as exc:
        components["withheld_reasons"].append(f"{type(exc).__name__}: {exc}")
    result["reward"] = scalar_reward(components, **config)
    return result


async def _request(job, contract, judge, result):
    method = job["method"]
    request = {
        "method": method,
        "model": "gpt-6.1-sol",
        "reasoning_effort": contract["judge"]["reasoning_effort"],
        "system": contract["judge"]["systems"][method],
        "payload": job["payload"],
    }
    result["judge_requests"].append(request)
    response = await judge(request)
    result["judge_responses"].append(response)
    return response


def response_schema(method):
    return MatchResponse if method == "sgr_match" else _grader().SCHEMAS["sgr_reason"]
