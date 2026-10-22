"""Free-text Solidity benchmark: the model audits in its own words; format earns nothing.

The policy prompt sets no output format. One judge call extracts the vulnerabilities a report
asserts, without adding or rewording claims; the reviewed one-to-one matcher then grades them
against the private ground truth, so coverage stays comparable with earlier suites.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_matching import Judge, evaluate_findings
from grpo.placement import rollout_gpus
from runtime_io import read_json

SUITE = "solidity_free_text"
SHA256 = re.compile(r"[0-9a-f]{64}")
SEVERITIES = ("High", "Medium", "Low", "Info", "N/A")
FINDING_KEYS = {"issue", "description", "severity", "contracts"}
SCOPE_PATH = Path(__file__).parents[1] / "benchmark_scope.json"
# Settings an evaluation config copies unchanged from the reviewed contract.
PINNED_KEYS = (
    "benchmark_suite",
    "model_id",
    "model_revision",
    "base_precision",
    "native_adapter_precision",
    "rollout_adapter_precision",
    "sampler_backend",
    "rollout_image",
    "native_image",
    "evaluation_kind",
    "benchmark_judge",
)


def audit_prompt(sources: str) -> str:
    return f"""You are a smart contract security expert. Audit the contracts below: analyse
their logic and purpose and find every security vulnerability you can.

Write your findings in whatever way you find clearest. For each vulnerability, explain where
it is, its root cause, how it can be triggered and what its impact is.

{sources}
"""


def extraction_prompt(report: str) -> str:
    return f"""Below is a security report written by an auditor, in free text. List every
distinct vulnerability the auditor asserts is present in the code.

Rules:
- Include only vulnerabilities the auditor concludes are real. Leave out ideas the auditor
  considered and rejected, open questions and general advice.
- Do not add, merge, split, correct or strengthen claims. One entry per vulnerability as the
  auditor describes it.
- "issue": a short title in the auditor's terms. "description": the auditor's own explanation
  of location, root cause, trigger and impact, copied or closely paraphrased.
- "severity": the auditor's stated severity (High, Medium, Low or Info), or "N/A" if none.
- "contracts": the files the auditor names for the vulnerability, or [] if none.
- If the auditor reports no vulnerabilities, return an empty list.

## Report
{report}
"""


# --- Reviewed configuration


def evaluation_row_ids() -> list[str]:
    """The benchmark: the five Solidity repositories every evaluation runs."""
    return read_json(SCOPE_PATH)["row_ids"]


def validate_config(config: dict[str, Any]) -> None:
    scope = read_json(SCOPE_PATH)
    allowed_rows = scope["row_ids"] + (
        scope["extra_verification_row_ids"] if config.get("extra_verification") is True else []
    )
    if config.get("row_id") not in allowed_rows:
        raise ValueError(f"Project {config.get('row_id')!r} is outside {allowed_rows}")
    for key in PINNED_KEYS:
        if config.get(key) != scope[key]:
            raise ValueError(f"Free-text benchmark differs from reviewed contract: {key}")
    if config.get("output_format") != "free_text":
        raise ValueError("The free-text suite sets no output format")
    effort = config.get("template_options", {}).get("reasoning_effort")
    if effort not in scope["reasoning_efforts"]:
        raise ValueError(f"Reasoning effort {effort!r}; reviewed: {scope['reasoning_efforts']}")
    if config["template_options"] != dict(scope["template_options"], reasoning_effort=effort):
        raise ValueError("Chat template options differ from the reviewed contract")
    context = scope["engine"]["max_model_len"]
    prompt_tokens = config.get("prompt_token_count")
    if type(prompt_tokens) is not int or not 0 < prompt_tokens < context:
        raise ValueError("Measured prompt token count required")
    if config.get("sampling") != dict(scope["sampling"], max_tokens=context - prompt_tokens):
        raise ValueError("Sampling must be the reviewed draws over the remaining context")
    actual = config.get("engine", {})
    rollout_gpus(actual)  # Raises for an unreviewed tensor-parallel size.
    # Tensor parallelism is a host placement choice, checked by rollout_gpus above.
    if {**actual, "tensor_parallel_size": None} != dict(scope["engine"], tensor_parallel_size=None):
        raise ValueError("Engine differs from the reviewed full-context contract")
    for key in ("seconds_limit", "judge_seconds_limit", "judge_operation_cap_usd"):
        value = config.get(key)
        if type(value) not in (int, float) or not 0 < value <= scope[key]:
            raise ValueError(f"{key} must be in (0, {scope[key]}]")
    if type(config.get("judge_cap_usd")) not in (int, float) or not config["judge_cap_usd"] > 0:
        raise ValueError("A positive judge ledger cap is required")
    if (
        config.get("grading_concurrency") not in (1, 4)
        or type(config.get("overlap_grading")) is not bool
    ):
        raise ValueError("Grading concurrency 1 or 4 and an explicit overlap flag required")
    validate_cases(config)


def validate_cases(config: dict[str, Any]) -> None:
    cases = config.get("cases", [])
    reference = config.get("base_reference")
    if reference is not None:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"run_root", "comparison_sha256"}
            or not isinstance(reference["run_root"], str)
            or not SHA256.fullmatch(str(reference["comparison_sha256"]))
        ):
            raise ValueError("The frozen base reference names its run root and comparison hash")
        if len(cases) != 1 or cases[0].get("label") == "base":
            raise ValueError("Against a frozen base, the run generates one trained arm only")
        trained = cases
    elif not cases or cases[0] != {"label": "base", "initial_step": 0} or len(cases) > 2:
        raise ValueError(
            "Cases are the base arm, optionally followed by one trained arm, "
            "or one trained arm with a frozen base reference"
        )
    else:
        trained = cases[1:]
    for case in trained:
        step = case.get("initial_step")
        if (
            type(step) is not int
            or case.get("label") != f"checkpoint-{step}"
            or not case.get("checkpoint")
            or not case.get("initial_adapter_sha256")
        ):
            raise ValueError("A trained arm names its step, checkpoint and adapter hash")
        if not config.get("merged_model_path"):
            raise ValueError("A trained arm is evaluated through its merged weights")
    if sorted(config.get("generation_order", [])) != sorted(case["label"] for case in cases):
        raise ValueError("Generation order must list every arm once")


# --- Grading


def validate_extracted(report: Any) -> list[dict[str, Any]]:
    if not isinstance(report, dict) or set(report) != {"findings"}:
        raise ValueError("Extraction must be an object with only 'findings'")
    findings = report["findings"]
    if not isinstance(findings, list):
        raise ValueError("Extracted findings must be a list")
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != FINDING_KEYS:
            raise ValueError(f"Extracted finding keys must be {sorted(FINDING_KEYS)}")
        if any(not isinstance(finding[key], str) for key in ("issue", "description")):
            raise ValueError("Extracted issue and description must be text")
        if finding["severity"] not in SEVERITIES:
            raise ValueError(f"Extracted severity must be one of {SEVERITIES}")
        if not isinstance(finding["contracts"], list) or any(
            not isinstance(path, str) for path in finding["contracts"]
        ):
            raise ValueError("Extracted contracts must be a list of paths")
    return findings


async def grade_draw(
    sample: dict[str, Any],
    truth: list[dict[str, Any]],
    template: str,
    extract: Judge,
    match: Judge,
) -> dict[str, Any]:
    """Grade one draw's own words; a cut-off or empty answer stays unknown."""
    text = sample["final_text"].strip()
    unknown = {
        "known_gt_count": len(truth),
        "exact_coverage": None,
        "coverage_including_partial": None,
    }
    if sample["finish_reason"] == "length":
        return {"status": "truncated", **unknown}
    if sample["finish_reason"] != "stop" or not text:
        return {"status": "no_answer", **unknown}
    # A draw that stopped inside its reasoning is graded on what that reasoning concludes.
    source = (
        "final_answer" if "</think>" in sample.get("raw_text", "</think>") else "unclosed_reasoning"
    )
    findings = validate_extracted(await extract(extraction_prompt(text)))
    result = await evaluate_findings(findings, truth, template, match)
    return {**result, "answer_source": source, "extracted_findings": findings}


def summarize(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm means over graded draws, every status counted, and deltas versus the base."""
    summary = {}
    complete = True
    for arm in arms:
        draws = arm["draws"]
        graded = [row for row in draws if row["status"] == "graded"]
        complete = complete and len(graded) == len(draws)
        summary[arm["label"]] = {
            "status_counts": dict(Counter(row["status"] for row in draws)),
            "mean_exact_coverage": mean(row["exact_coverage"] for row in graded),
            "mean_coverage_including_partial": mean(
                row["coverage_including_partial"] for row in graded
            ),
            "mean_completion_tokens": mean(row["completion_tokens"] for row in draws),
        }
    result = {
        "status": "complete" if complete else "incomplete",
        "arms": summary,
        "quality_gain_established": False,
    }
    if complete and len(arms) == 2:
        base, trained = (summary[arm["label"]] for arm in arms)
        result["deltas"] = {
            arms[1]["label"]: {
                key: trained[key] - base[key]
                for key in ("mean_exact_coverage", "mean_coverage_including_partial")
            }
        }
    return result


def mean(values: Any) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None
